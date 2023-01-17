!pip uninstall -y typing
!pip install git+https://github.com/thomasbrandon/l5kit@v1.0.6-perf#subdirectory=l5kit
!pip install omegaconf
from typing import List, Optional, Tuple

from l5kit.data import DataManager, LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer, Rasterizer, SemanticRasterizer, BoxRasterizer, SemBoxRasterizer, StubRasterizer
from l5kit.rasterization.rasterizer_builder import _load_metadata

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import numba as nb
from time import perf_counter
from tqdm.auto import tqdm
from contextlib import contextmanager
import importlib
CONFIG_STR = """
# Config format schema number
format_version: 4

###################
## Model options
model_params:
  history_num_frames: 0
  history_step_size: 1
  history_delta_time: 0.1

  future_num_frames: 50
  future_step_size: 1
  future_delta_time: 0.1

###################
## Input raster parameters
raster_params:
  # raster image size [pixels]
  raster_size:
    - 224
    - 224
  # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
  pixel_size:
    - 0.5
    - 0.5
  # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
  ego_center:
    - 0.25
    - 0.5
  map_type: "py_semantic"

  # the keys are relative to the dataset environment variable
  satellite_map_key: "aerial_map/aerial_map.png"
  semantic_map_key: "semantic_map/semantic_map.pb"
  dataset_meta_key: "meta.json"

  # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
  # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
  filter_agents_threshold: 0.5

###################
## Data loader options
train_data_loader:
  key: "scenes/train.zarr"
"""
CONFIG =  OmegaConf.create(CONFIG_STR)

DATA_DIR="/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager(DATA_DIR)
def create_dataset(cfg, zarr, map_type="SemBox", raster_size=None, pixel_size=None, rast_class=None):
    if not map_type is None: cfg.raster_params['map_type'] = map_type
    if not raster_size is None: cfg.raster_params['raster_size'] = [raster_size, raster_size]
    if not pixel_size is None: cfg.raster_params['pixel_size'] = [pixel_size, pixel_size]
        
    # Build rasterizer
    raster_size = tuple(cfg.raster_params.raster_size)
    pixel_size = np.array(cfg.raster_params.pixel_size)
    ego_center = np.array(cfg.raster_params.ego_center)
    semantic_map_filepath = dm.require(cfg.raster_params.semantic_map_key)
    dataset_meta = _load_metadata(cfg.raster_params.dataset_meta_key, dm)
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
    filter_agents_threshold = cfg.raster_params.filter_agents_threshold
    history_num_frames = cfg.model_params.history_num_frames
    if map_type == "SemBox":
        if rast_class is None: rast_class = SemBoxRasterizer
        args = (raster_size, pixel_size, ego_center, filter_agents_threshold, history_num_frames, semantic_map_filepath, world_to_ecef,)
    elif map_type == "Semantic":
        if rast_class is None: rast_class = SemanticRasterizer
        args = (raster_size, pixel_size, ego_center, semantic_map_filepath, world_to_ecef,)
    elif map_type == "Box":
        if rast_class is None: rast_class = BoxRasterizer
        args = (raster_size, pixel_size, ego_center, filter_agents_threshold, history_num_frames)
    else:
        raise ValueError("Unknown rasterizer type: " + map_type)
    rast = rast_class(*args)
    # Build dataset
    ds = AgentDataset(cfg, zarr, rast)
    return ds

def fmt_time(val):
  units = ('p','n','µ','m','','k')
  scale = int(np.floor(np.log10(np.abs(val))/3)) if val != 0 else 0
  if scale < 2:
    val *= 1000**(-scale)
    unit = units[scale+4]
    return f'{val:.1f}{unit}s'
  return f"{val:.1g}s"

def summarise_times(times: np.array, iter_sec:bool=False):
    summ = f"{fmt_time(times.mean())} +/- {fmt_time(times.std())}"
    if iter_sec: summ += f"; {len(times)/times.sum():.1f}it/s"
    return summ

def dataset_perf(ds, num=500, summary=False, report=True, progress=True):
    n = len(ds)
    times = [0] * num
    it = range(num)
    if progress: it = tqdm(it)
    for i in it:
        start = perf_counter()
        # Don't just get sequential items to pull from different scenes
        _ = ds[i * 9773 % n]
        times[i] = perf_counter() - start
    times = np.array(times)
    summ = summarise_times(times, iter_sec=True)
    if report:
        print(f"Retrieved {num} items in {fmt_time(times.sum())}.\n  {summ}")
    if summary:
        return summ
    else:
        return times
data_path = dm.require(CONFIG.train_data_loader.key)
train_zarr = ChunkedDataset(data_path).open()
train_ds = create_dataset(CONFIG, train_zarr, map_type="Semantic")
print(train_ds)
# Transform an array of points with a coordinate system transformation matrix
# For d dimensional points the transformation matrix should be of size d+1 (i.e. 2D points use a 3x3 matrix).
@nb.guvectorize([(nb.float64[:,:], nb.float64[:,:], nb.float64[:,:])],
                "(p,d),(t,t)->(p,d)", nopython=True)
def transform_points_nb(points, transf_matrix, res):
    n_dim = transf_matrix.shape[0] - 1
    #assert points.shape[1] == n_dim, "Mismatched dimensions"
    # For each point compute a dot product with the transformation matrix fixing the Z coord to 1.
    for p in range(points.shape[0]):
        for out_dim in range(n_dim):
            val = 0
            for dim in range(n_dim):
                val += points[p, dim] * transf_matrix[out_dim, dim]
            val += transf_matrix[out_dim, n_dim] # *1 - Fixed Z
            res[p,out_dim] = val
CV2_SHIFT_VALUE = 256
# Transform an array of points with a coordinate system transformation matrix
# For d dimensional points the transformation matrix should be of size d+1 (i.e. 2D points use a 3x3 matrix).
@nb.guvectorize([(nb.float64[:,:], nb.float64[:,:], nb.int32[:,:])],
                "(p,d),(t,t)->(p,d)", nopython=True)
def transform_points_subpixel_nb(points, transf_matrix, res):
    n_dim = transf_matrix.shape[0] - 1
    #assert points.shape[1] == n_dim, "Mismatched dimensions"
    # For each point compute a dot product with the transformation matrix fixing the Z coord to 1.
    for p in range(points.shape[0]):
        for out_dim in range(n_dim):
            val = 0
            for dim in range(n_dim):
                val += points[p, dim] * transf_matrix[out_dim, dim]
            val += transf_matrix[out_dim, n_dim] # *1 - Fixed Z
            res[p,out_dim] = int(val * CV2_SHIFT_VALUE)
from l5kit.geometry import transform_points
from l5kit.rasterization.semantic_rasterizer import transform_points_subpixel
test_item = train_ds[0]
transf_matrix = test_item["world_to_image"]
transf_matrix
lane_id = train_ds.rasterizer.bounds_info["lanes"]["ids"][0]
lane_coords = train_ds.rasterizer.proto_API.get_lane_coords(lane_id)
test_points = lane_coords['xyz_left'][:,:2]
test_points[:5]
np.testing.assert_equal(transform_points(test_points, transf_matrix),
                        transform_points_nb(test_points, transf_matrix))
np.testing.assert_equal(transform_points_subpixel(test_points, transf_matrix),
                        transform_points_subpixel_nb(test_points, transf_matrix))
def transform_perf(func, transf_matrix, num_items=100, num_repeats=1000, num_warmup=100, report=False):
    inps = np.random.randn(num_repeats+num_warmup, num_items, 2)
    times = [0] * inps.shape[0]
    for i in range(inps.shape[0]):
        it = inps[i, :, :]
        start = perf_counter()
        res = func(it, transf_matrix)
        times[i] = perf_counter() - start
    times = np.array(times[num_warmup:])
    if report: print(summarise_times(times))
    return times
times = transform_perf(transform_points, transf_matrix, report=True)
results = []
funcs = {"transform_points": (transform_points, transform_points_nb),
         "transform_points_subpixel": (transform_points_subpixel, transform_points_subpixel_nb)}
for func_name, (orig_func,numba_func) in funcs.items():
    print("Testing " + func_name)
    for num_items in (10, 1000, 10000, 100000):
        orig_times = transform_perf(orig_func, transf_matrix, num_items)
        numba_times = transform_perf(numba_func, transf_matrix, num_items)
        results.append({"Function": func_name, "Points": num_items,
                        "Original": summarise_times(orig_times),
                        "Numba": summarise_times(numba_times),
                        "orig_mean": orig_times.mean(), "numba_mean": numba_times.mean()})
results = pd.DataFrame(results)
results["Improvement"] = (1-(results.numba_mean/results.orig_mean)).apply("{:.0%}".format)
results[["Function","Points","Original","Numba","Improvement"]]
@contextmanager
def create_patched_dataset(*args, **kwargs):
    mod = importlib.import_module("l5kit.rasterization.semantic_rasterizer")
    orig_func = mod.transform_points_subpixel
    mod.transform_points_subpixel = transform_points_subpixel_nb
    yield create_dataset(*args, **kwargs, rast_class=mod.SemanticRasterizer)
    mod.transform_points_subpixel = orig_func
orig_times = dataset_perf(train_ds)
with create_patched_dataset(CONFIG, train_zarr, map_type="Semantic") as numba_ds:
    numba_times = dataset_perf(numba_ds)
print(f"\n{1-(numba_times.mean() / orig_times.mean()):.1%} improvement")
