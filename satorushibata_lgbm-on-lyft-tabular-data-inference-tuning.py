import pandas as pd, numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
%matplotlib inline
import itertools as it
from pathlib import Path

pd.options.display.max_columns=305
import lightgbm  as lgb
import gc
from sklearn.model_selection import train_test_split
import time
from tqdm.notebook import tqdm

import re,json
DATA_ROOT = Path("../input/lyft-train-as-parquet/train")
def get_scene_path(scene):
    meta = "meta_{}_{}.json".format(*re.search( r"scenes_(\d+)_(\d+)", scene.stem).groups())
    with open(DATA_ROOT/meta) as f:
        meta = json.load(f)
    frame = DATA_ROOT/meta["frames"]["results"]["filename"]
    agent = DATA_ROOT/meta["agents"]["results"]["filename"]
    return (scene, frame, agent)
SCENES = np.array(list(DATA_ROOT.glob("scenes_*.parquet.snappy")))
SCENES = SCENES[np.random.permutation(len(SCENES))]
print("NB SCENES:", len(SCENES))
scene = SCENES[0]
scene
get_scene_path(scene)
reader = pd.read_parquet
def merge(scenes, frames, agents, shift, verbose=False):
    df = scenes.merge(frames, on = "scene_db_id")
    df = df.merge(agents, on="frame_db_id")

    shift_cols = [
            "centroid_x", "centroid_y",
            "yaw",
            "velocity_x","velocity_y",
            "nagents","nlights",
            'extent_x', 'extent_y','extent_z',
            'label_probabilities_PERCEPTION_LABEL_UNKNOWN',
            'label_probabilities_PERCEPTION_LABEL_CAR', 
            'label_probabilities_PERCEPTION_LABEL_CYCLIST', 
            'label_probabilities_PERCEPTION_LABEL_PEDESTRIAN',
    ]
    new_shift_cols = ["centroid_xs", "centroid_ys"] + shift_cols[2:]

    df[new_shift_cols] = df.groupby(["scene_db_id", "track_id"])[shift_cols].shift(shift)
    nulls = df[["centroid_xs", "centroid_ys", "yaw", "velocity_x","velocity_y"]].isnull().any(1)
    shape0 =  df.shape
    df = df[~nulls]
    
    if verbose:
        print("SHAPE0:", shape0)
        print("nulls ratio:", nulls.sum()/shape0[0])
    
    return df
def read_all(shift=1, max_len=1e5):
    """
    Read parquet files from the SCENES list until the df's size is greater than `max_len`.
    
    If you want better accuracy, you need to increase `max_len`.
    With `max_len=12e6` I got a score of 200.xxx.
    But note that training time increases as max_len increases.
    """
    dfs = None
    scenes = SCENES[np.random.permutation(len(SCENES))]
    for scene in scenes:
        SCENES_FILE,FRAMES_FILE,AGENTS_FILE = get_scene_path(scene)
        
        scenes = reader(SCENES_FILE)
        
        frames = reader(FRAMES_FILE)
        frames["nagents"] = frames["agent_index_interval_end"] - frames["agent_index_interval_start"]
        frames["nlights"] = frames["traffic_light_faces_index_interval_end"
                                  ] - frames["traffic_light_faces_index_interval_start"]
    
        agents = reader(AGENTS_FILE)
        agents.rename(columns = {"agent_id": "agent_db_id"}, inplace=True)
        
        df = merge(scenes, frames, agents, shift=shift)
        
        dfs = df if dfs is None else dfs.append(df)
        dfs.reset_index(inplace=True, drop=True)
        
        if len(dfs) > max_len:
            break
    
    return dfs
def lgbm_trainer(shift=1, root=None, params=None):
    t0 = time.strftime("%Y%m%d%H%M%S")
    T00 = time.time()
    root = "model_{}".format(t0) if root is None else str(root)
    params = PARAMS if params is None else params
    
    df = read_all(shift=shift)
    print("df.shape:", df.shape)
    
    df_centroid = df[["centroid_x", "centroid_y"]]
    df = df[TRAIN_COLS]
    gc.collect()
    
    train_index, test_index = train_test_split(df.index.values.reshape((-1,1)),
                                           df.index.values, test_size = .20, random_state=177)[2:]
    print("\n")
    for suffix in ["x", "y"]:
        print("--> {}".format(suffix.upper()))
        target_name = "centroid_" + suffix
        target = (df_centroid[target_name] - df[target_name+"s"])
    
        train_data = lgb.Dataset(df.loc[train_index], label= target.loc[train_index])
        test_data = lgb.Dataset(df.loc[test_index], label= target.loc[test_index])

        clf = lgb.train(params,
                        train_data,
                        valid_sets = [train_data, test_data],
                        early_stopping_rounds=60, 
                        verbose_eval= 40
                       )
        
        clf.save_model("models/{}/lgbm_{}_shift_{:02d}.bin".format(root, suffix, shift))
        print('\n')
    print("elapsed: {:.5f} min".format((time.time()-T00)/60))
def get_time(format_="%Y-%m-%d %H:%M:%S"):
    return time.strftime(format_)
def train_50_shifts(root=None):
    root = root or  "model_{}".format(time.strftime("%Y%m%d%H%M%S"))
    Path("models").joinpath(root).mkdir(exist_ok=True, parents=True)
    params = PARAMS.copy()
    for shift in tqdm(list(range(50, 0, -1))):
        print('\n ******************* SHIFT {:02d} {} ***********\n'.format(shift,get_time()))
        
        if not (shift-1)%5:
            params["num_iterations"] = max(100, params["num_iterations"] - 20)
            params["num_leaves"] = max(31, params["num_leaves"] - 10)
        
        meta = {
            "TRAIN_COLS": TRAIN_COLS,
            "params": params,
            "shift": shift,
            "start": get_time(),
            "end": None
        }
        
            
        lgbm_trainer(root=root, shift=shift, params=params)
        meta["end"] = get_time()
        with open("models/{}/meta_shift_{:02d}.json".format(root, shift), "w") as f:
            json.dump(meta, f, indent=2)
PARAMS = {
         'objective':'regression',
         'boosting': 'gbdt',
         'feature_fraction': 0.5 ,
         'scale_pos_weight' : 1/40., 
         'num_iterations' : 200,
         'learning_rate' :  0.7,
         'max_depth': 41,
         'min_data_in_leaf': 64,
         'num_leaves': 128,
         'bagging_freq' : 1,
         'bagging_fraction' : 0.8,
         'tree_learner': 'voting',
         'boost_from_average': True,
         'verbosity' : 0,
         'num_threads': 2,
         'metric' : ['mse'],
         'metric': [ "l1", "rmse"],
         "verbosity": 1,
         'reg_alpha': 0.1,
         'reg_lambda': 0.3
        }
# Uncomment the columns if you want more

TRAIN_COLS = [
     'ego_translation_x', 
     'ego_translation_y', 
     'ego_translation_z', 
     'ego_rotation_xx', 
     'ego_rotation_xy', 
     'ego_rotation_xz', 
     'ego_rotation_yx', 
     'ego_rotation_yy', 
     'ego_rotation_yz', 
     'ego_rotation_zx', 
     'ego_rotation_zy', 
     'ego_rotation_zz', 
    'extent_x', 
    'extent_y', 
    'extent_z', 
    'velocity_x', 
    'velocity_y', 
    'label_probabilities_PERCEPTION_LABEL_UNKNOWN', 
    'label_probabilities_PERCEPTION_LABEL_CAR', 
    'label_probabilities_PERCEPTION_LABEL_CYCLIST', 
    'label_probabilities_PERCEPTION_LABEL_PEDESTRIAN', 
    'yaw', 
    'nagents', 
    'nlights', 
    'centroid_xs', 
    'centroid_ys',
]
print("len(TRAIN_COLS):", len(TRAIN_COLS))
%%time

# Train 50x2 lgbm models (50 time dimensions X 2 space dimensions)
# Save it as lgbm_{x or y}_shift_{i:02d}
# Each model has it's own meta_shift_{i:02d} which contains the model's params
# You can juts feed the ouputs as inputs to the inference kernel
# The inference kernel is at https://www.kaggle.com/kneroma/lgbm-on-lyft-tabular-data-inference

train_50_shifts("lyft_lgbm_model")
# Here, I'm gonna load the test, it contains `71122` rows as expected
df = pd.read_csv("../input/lyft-test-set-as-csv/Lyft_test_set.csv")
print("df.shape:", df.shape)
df.head(10)
def get_model_name(filename):
    return re.search("^(lgbm_[x,y]_shift_\d+)", filename).group(1)
def get_models(path):
    models = {}
    path = Path(path)
    for model in path.glob("lgbm*"):
        model_name = get_model_name(model.stem)
        shift = int(model_name.split("shift_")[1])
        meta = path.joinpath("meta_shift_{:02d}.json".format(shift))
        with meta.open() as f:
            train_cols = json.load(f)["TRAIN_COLS"]
        models[model_name] = {"model": model.as_posix(), "train_cols": train_cols}
    return models
models = get_models("./models/lyft_lgbm_model")
len(models)
next(iter(models.items()))
def make_colnames():
    xcols = ["coord_x{}{}".format(step, rank) for step in range(3) for rank in range(50)]
    ycols = ["coord_y{}{}".format(step, rank) for step in range(3) for rank in range(50)]
    cols = ["timestamp", "track_id"] + ["conf_0", "conf_1", "conf_2"] + list(it.chain(*zip(xcols, ycols)))
    return cols
def predict(models, df):
    sub = np.empty((len(df), 305))
    sub.fill(np.nan)
    sub = pd.DataFrame(sub, columns = make_colnames())
    sub[["timestamp", "track_id"]] = df[["timestamp", "track_id"]]
    sub["conf_0"] = 1.0
    
    for shift in range(1, 51):
        for suffix in ["x", "y"]:
            model_info = models["lgbm_{}_shift_{:02d}".format(suffix, shift)]
                
            model = lgb.Booster(model_file= model_info["model"])
            pred = model.predict(df[model_info["train_cols"]])
            
            sub["coord_{}0{}".format(suffix, shift-1)] = pred

        if not shift%10:
            print("shift: {}".format(shift))
    
    sub.fillna(0., inplace=True)
    
    return sub
sub = predict(models, df)
sub.iloc[:50, :105]
sub.to_csv("submission.csv", index=False)