import numpy as np 

import pandas as pd 

from fastai.vision import *

import geopandas as gpd

import shapely

from shapely.geometry import Polygon, LineString

from tqdm import tqdm
path = Path('/kaggle/input/global-wheat-detection/')
df = pd.read_csv(path/'train.csv')

df
def bbox2mask(x):

    labels = np.array(x)

    mask = torch.zeros(1024,1024)

    for l in labels:

        mask[l[1]:l[1]+l[3], l[0]:l[0]+l[2]] = 1

    return mask



def bbox_center(x):

    labels = np.array(x)

    mask = torch.zeros(1024,1024)

    for l in labels:

        mask[(2*l[1]+l[3])//2, (2*l[0]+l[2])//2] = 1

    return mask



def box2polygon(x):

    return Polygon([(x[0], x[1]), (x[0]+x[2], x[1]), (x[0]+x[2], x[1]+x[3]), (x[0], x[1]+x[3])])
boxes = df.groupby('image_id').agg({'bbox' : lambda x : list(x)})

box = boxes.iloc[2]

file = str(path/'train'/box.name) + '.jpg'

img = open_image(file).data.numpy().transpose(1,2,0)

bbox = np.array([eval(l) for l in box.bbox]).astype(int).tolist()

mask = bbox2mask(bbox)

gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in bbox]})

gdf.head()
def slice_box(box_A:Polygon, box_B:Polygon, margin=10, line_mult=10):

    "Returns box_A sliced according to the distance to box_B."

    vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])

    vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])

    vec_AB_norm = np.linalg.norm(vec_AB)

    split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)*margin

    line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])

    split_box = shapely.ops.split(box_A, line)

    if len(split_box) == 1: return split_box, None, line

    is_center = [s.contains(box_A.centroid) for s in split_box]

    if sum(is_center) == 0: 

        warnings.warn('Polygon do not contain the center of original box, keeping the first slice.')

        return split_box[0], None, line

    where_is_center = np.argwhere(is_center).reshape(-1)[0]

    where_not_center = np.argwhere(~np.array(is_center)).reshape(-1)[0]

    split_box_center = split_box[where_is_center]

    split_box_out = split_box[where_not_center]

    return split_box_center, split_box_out, line
inter = gdf.loc[gdf.intersects(gdf.iloc[20].geometry)]



box_A = inter.iloc[0].values[0]

box_B = inter.iloc[1].values[0]

polyA, _, lineA = slice_box(box_A, box_B, margin=10, line_mult=1.2)

polyB, _, lineB = slice_box(box_B, box_A, margin=10, line_mult=1.2)



boxes = gpd.GeoDataFrame({'geometry': [box_A, box_B]})

centroids =  gpd.GeoDataFrame({'geometry': [box_A.centroid, box_B.centroid]})

splited_boxes = gpd.GeoDataFrame({'geometry': [polyA, polyB]})

lines = gpd.GeoDataFrame({'geometry': [lineA, lineB]})



fig, ax = plt.subplots(dpi=120)

boxes.plot(ax=ax, facecolor='gray', edgecolor='k', alpha=0.5)

centroids.plot(ax=ax, c='k')

ax.axis('off');



fig, ax = plt.subplots(dpi=120)

boxes.plot(ax=ax, facecolor='gray', edgecolor='k', alpha=0.1)

splited_boxes.plot(ax=ax, facecolor='olive', edgecolor='k')

centroids.plot(ax=ax, c='k')

lines.plot(ax=ax, color='k')

ax.axis('off');
def intersection_list(polylist):

    r = polylist[0]

    for p in polylist:

        r = r.intersection(p)

    return r

    

def slice_one(gdf, index):

    inter = gdf.loc[gdf.intersects(gdf.iloc[index].geometry)]

    if len(inter) == 1: return inter.geometry.values[0]

    box_A = inter.loc[index].values[0]

    inter = inter.drop(index, axis=0)

    polys = []

    for i in range(len(inter)):

        box_B = inter.iloc[i].values[0]

        polyA, *_ = slice_box(box_A, box_B)

        polys.append(polyA)

    return intersection_list(polys)



def slice_all(gdf):

    polys = []

    for i in range(len(gdf)):

        polys.append(slice_one(gdf, i))

    return gpd.GeoDataFrame({'geometry': polys})
res_df = slice_all(gdf)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5), dpi=120)

gdf.plot(ax=ax1, alpha=0.5, color='gray')

#gdf.plot(ax=ax2, alpha=0.1, facecolor='gray')

res_df.plot(ax=ax2, alpha=0.5, color='olive')

ax1.axis('equal')

ax2.axis('equal')

ax1.set_title('Original bounding boxes')

ax2.set_title('Splited bounding boxes')

fig.tight_layout()
import rasterio.features
raster = rasterio.features.rasterize(res_df.geometry, out_shape=(1024,1024), merge_alg=rasterio.enums.MergeAlg.replace)
fig, axes = plt.subplots(ncols=2, dpi=120)

axes[0].imshow(img)

axes[0].imshow(mask, alpha=0.4)

axes[1].imshow(img)

axes[1].imshow(raster, alpha=0.4)
import PIL

import zipfile

import cv2
mask = cv2.imencode('.png', (raster*255).astype(np.uint8))[1]
boxes = df.groupby('image_id').agg({'bbox' : lambda x : list(x)})



with zipfile.ZipFile('split_masks.zip', 'w') as mask_out:

    for i in progress_bar(range(len(boxes))):

        box = boxes.iloc[i]

        file = str(path/'train'/box.name) + '.jpg'

        img = open_image(file).data.numpy().transpose(1,2,0)

        bbox = np.array([eval(l) for l in box.bbox]).astype(int).tolist()

        gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in bbox]})

        res_df = slice_all(gdf)

        raster = rasterio.features.rasterize(res_df.geometry, out_shape=(1024,1024), merge_alg=rasterio.enums.MergeAlg.replace)

        mask = cv2.imencode('.png', (raster*255).astype(np.uint8))[1]

        mask_out.writestr(f'{box.name}.png', mask)