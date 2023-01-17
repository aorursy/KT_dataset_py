import bq_helper
from bq_helper import BigQueryHelper

nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                               dataset_name="new_york")

query = """SELECT
  PICKUP_LONGITUDE,
  PICKUP_LATITUDE
FROM
  `bigquery-public-data.new_york.tlc_yellow_trips_2016`
LIMIT 2000000;
"""

df = nyc.query_to_pandas_safe(query, max_gb_scanned=10)
df = df.rename(columns={"PICKUP_LONGITUDE":"lng", "PICKUP_LATITUDE":"lat" }) # Rename column
df.dropna(inplace=True) # Remove NaN data

b_bound, t_bound = 39.268981, 42.624965 # Latitude range
l_bound, r_bound = -76.616306, -71.182014 # Longtitude range
df = df[(l_bound <= df.lng) & (df.lng <= r_bound) & (b_bound <= df.lat) & (df.lat <= t_bound)] # Remove out range

df.head(5)
import os, glob, pickle, gzip
import mercantile
import concurrent.futures as futures
import datashader as ds
import pandas as pd

from mercantile import Tile

base_zoom = 12 # base zoom of map tile
agg_root = os.path.join("map", "agg")
tile_root = os.path.join("map", "tile")
def convGpsToWebMecator(row):
    lng, lat = row.lng, row.lat
    x, y = ds.utils.lnglat_to_meters(lng, lat)
    xtile, ytile, zoom = mercantile.tile(lng, lat, base_zoom)
    return (x, y, zoom, xtile, ytile)

df['x'], df['y'], df['zoom'], df['xtile'], df['ytile'] = zip(*df.apply(convGpsToWebMecator, axis=1))
df.head(5)
# Get corresponding path of aggregation file 
def getAggFilePath(root, x, y, z):
    return os.path.join(root, f'{z}', f'{x}', f'{y}.pkl.gz') 

# 使用 Pickle 序列化 Aggregation 並儲存成  gzip 格式的檔案
def serializeAggToFile(agg, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, mode='wb') as file:
        pickle.dump(agg, file)
        
# 依照 Tile 位置建立 datashader.Canvas
def mapTileCanvas(xtile, ytile, zoom, tile_size=(256, 256)):
    bounds = mercantile.xy_bounds(xtile, ytile, zoom)
    canvas = ds.Canvas(plot_width = tile_size[0],
                       plot_height = tile_size[1],
                       x_range = (bounds.left, bounds.right),
                       y_range = (bounds.bottom, bounds.top))
    return canvas
for ((zoom, xtile, ytile), data) in df.groupby(by=['zoom', 'xtile', 'ytile']):
    agg = mapTileCanvas(xtile, ytile, zoom).points(data, 'x', 'y')
    serializeAggToFile(agg, getAggFilePath(agg_root, xtile, ytile, zoom))
# A dummy dataframe for generating aggregation object
dummy_df = pd.DataFrame.from_dict(data={'x':[0], 'y': [0]})

def getTileFromPath(path):
    sep = path.split(os.sep)
    if len(sep) < 3:
        raise ValueError("aggregation file can't convert to tile path")     
    return (int(sep[-2]), int(sep[-1].split('.')[0]), int(sep[-3]))

# Resize matrix by sum 2x2 region (ex. 256x256 -> 128x128)
def poolmat(m):
    return m[::2, ::2] + m[::2, 1::2] + m[1::2, 1::2] + m[1::2, ::2]

# Load aggregation file of map from file system (if no file, then create a empty aggreated result)
def makeTileAggegation(root, tile):
    file_path = getAggFilePath(root, *tile)
    if os.path.exists(file_path):
        try:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        except:
            return mapTileCanvas(*tile).points(dummy_df, 'x', 'y')
    else:
        return mapTileCanvas(*tile).points(dummy_df, 'x', 'y')
# Generate map tile by bottom-up method
def combineTileButtomUp(root, x, y, z):
    
    agg = mapTileCanvas(x, y, z).points(dummy_df, 'x', 'y')

    row, col = agg.values.shape
    row_c, col_c = int(row/2), int(col/2)
    
    # lt = left-top, rt = right-top, rb = right-bottom, lb = left-bottom
    lt, rt, rb, lb = mercantile.children(Tile(x, y, z)) 
    
    # Combine 
    agg.values[0:row_c, 0:col_c] = poolmat(makeTileAggegation(root, lb).values)
    agg.values[0:row_c, col_c:col] = poolmat(makeTileAggegation(root, rb).values)
    agg.values[row_c:row, col_c:col] = poolmat(makeTileAggegation(root, rt).values)
    agg.values[row_c:row, 0:col_c] = poolmat(makeTileAggegation(root, lt).values)
    
    return agg
# Generate list of parents tiles
def parentTiles(root, zoom):
    p_tile_set = set()
    files = glob.glob(os.path.join(root, str(zoom), "*", "*.pkl.gz"))
    for x, y, z in map(getTileFromPath, files):
        p_tile = mercantile.parent(Tile(x, y, z))
        p_tile_set.add((p_tile.x, p_tile.y, p_tile.z))
    return list(p_tile_set)

def chunks(datas, n):
    for i in range(0, len(datas), n):
        yield datas[i:i + n]
        
def chunksSize(num, factor):
    s = int(num / (os.cpu_count() * factor))
    return s if s > 0 else 1
# Paralled function
def makeTilesBottomUp(root, x, y, z):
    agg = combineTileButtomUp(root, x, y, z)
    serializeAggToFile(agg, getAggFilePath(root, x, y, z))

def makeTilesBottomUpWrapper(tuples):
    for tuple_obj in tuples:
        makeTilesBottomUp(*tuple_obj)
for zoom in range(base_zoom, 0, -1):
    with futures.ProcessPoolExecutor() as executor:        
        ptiles = [(agg_root, x, y, z) for x, y, z in parentTiles(agg_root, zoom)]        
        ptiles = list(chunks(ptiles, chunksSize(len(ptiles), 1)))
        fs = executor.map(makeTilesBottomUpWrapper, ptiles)
        futures.as_completed(fs)
import datashader.transfer_functions as tf
from colorcet import fire

def getRenderImage(img_root, agg_path):
    x, y, z = getTileFromPath(agg_path)
    return os.path.join(img_root, f'{z}', f'{x}', f'{y}.png')

def makeTileImage(source, target):
    with gzip.open(source, 'rb') as f:
        x, y, z = getTileFromPath(source)
        tile_path = os.path.join(tile_root, f'{z}', f'{x}', f'{y}.png')
        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
        
        agg = pickle.load(f)
        img = tf.shade(agg.where(agg > 5), cmap=fire)
        img = tf.set_background(img, color='black') # Set backgound 'black' to visualize tile image
        with open(tile_path, mode='wb') as out:
            out.write(img.to_bytesio(format='png').read())

def makeTileImageWrapper(tuple_list):
    for source, target in tuple_list:
        makeTileImage(source, target)
with futures.ProcessPoolExecutor() as executor:
    render_agg_list = glob.glob(os.path.join(agg_root, "*", "*", "*.pkl.gz"))
    render_img_list = [getRenderImage(tile_root, agg_path) for agg_path in render_agg_list]

    datas = list(zip(render_agg_list, render_img_list))  
    datas = list(chunks(datas, chunksSize(len(datas), 1)))
    
    fs = executor.map(makeTileImageWrapper, datas)
    futures.as_completed(fs)
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
list_of_files = ['map/tile/11/602/769.png', 'map/tile/11/603/769.png', 'map/tile/11/602/770.png', 'map/tile/11/603/770.png']

fig = figure()
for i in range(4):
    a = fig.add_subplot(2,2,i+1)
    image = imread(list_of_files[i])
    imshow(image)
    axis('off')
from IPython.display import HTML, IFrame
IFrame(src='https://rawgit.com/yeshuanova/nyc_taxi_trip_map/master/map.html',
       width=800, height=600)
