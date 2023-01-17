# import some useful packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from shapely.wkt import loads as wkt_loads

import matplotlib.pyplot as plt

import pylab

import json

import shapely.geometry

import os
# Input data files are available in the "../input/" directory.

inDir = '../input'

geojsonDir = 'train_geojson_v3'



# Give short names to object types

CLASSES = {

        1 : 'Bldg',

        2 : 'Struct',

        3 : 'Road',

        4 : 'Track',

        5 : 'Trees',

        6 : 'Crops',

        7 : 'Fast H20',

        8 : 'Slow H20',

        9 : 'Truck',

        10 : 'Car',

        }

CLASSES_R = {}

for k in CLASSES:

    CLASSES_R[CLASSES[k]] = k

COLORS = {

        1 : '0.7',

        2 : '0.4',

        3 : '#b35806',

        4 : '#dfc27d',

        5 : '#1b7837',

        6 : '#a6dba0',

        7 : '#74add1',

        8 : '#4575b4',

        9 : '#f46d43',

        10: '#d73027',

        }



def load_tables():

    # read the training data from train_wkt_v3.csv

    df = pd.read_csv(inDir + '/train_wkt_v3.csv',

            names=['SceneId', 'ClassType', 'MultipolygonWKT'], skiprows=1)



    # grid size will also be needed later..

    gs = pd.read_csv(inDir + '/grid_sizes.csv',

            names=['SceneId', 'Xmax', 'Ymin'], skiprows=1)

    return df, gs
df, gs = load_tables()

df.head()

gs.head()

allSceneIds = gs.SceneId.unique()

trainSceneIds = np.sort(df.SceneId.unique())
filename_to_classType = {

        '001_MM_L2_LARGE_BUILDING':1,

        '001_MM_L3_RESIDENTIAL_BUILDING':1,

        '001_MM_L3_NON_RESIDENTIAL_BUILDING':1,

        '001_MM_L5_MISC_SMALL_STRUCTURE':2,

        '002_TR_L3_GOOD_ROADS':3,

        '002_TR_L4_POOR_DIRT_CART_TRACK':4,

        '002_TR_L6_FOOTPATH_TRAIL':4,

        '006_VEG_L2_WOODLAND':5,

        '006_VEG_L3_HEDGEROWS':5,

        '006_VEG_L5_GROUP_TREES':5,

        '006_VEG_L5_STANDALONE_TREES':5,

        '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND':6,

        '007_AGR_L6_ROW_CROP':6, 

        '008_WTR_L3_WATERWAY':7,

        '008_WTR_L2_STANDING_WATER':8,

        '003_VH_L4_LARGE_VEHICLE':9,

        '003_VH_L5_SMALL_VEHICLE':10,

        '003_VH_L6_MOTORBIKE':10}

cType_to_Filename = {}

for cType in range(1,11):

    cType_to_Filename[cType] = [k for k in filename_to_classType if filename_to_classType[k] == cType]
# Basic functions to work with polygons

def get_grid_size(sceneId):

    '''

    Get the grid size of the scene



    Parameters

    ----------

    sceneId : str

        sceneId as used in grid_size.csv



    Returns

    -------

    (xmax, ymin) : float

    '''

    xmax, ymin = gs[gs.SceneId == sceneId].iloc[0,1:].astype(float)

    return xmax, ymin





def get_grid_area(sceneId):

    '''

    Get the area of the grid of the scene

    '''

    xmax, ymin = get_grid_size(sceneId)

    return np.abs(xmax*ymin)





def get_polygons(sceneId):

    '''

    Parameters

    ----------

    sceneId : str

        sceneId like "6010_0_4"



    Returns

    -------

    polygonsList : dict

        Keys are CLASSES

        Values are shapely polygons

    '''

    # df_scene = df[df.SceneId == sceneId]

    

    polygonsList = {}

    for cType in CLASSES.keys():

        # WKT version:

        # polygonsList[cType] = wkt_loads(df_scene[df_scene.ClassType == cType].MultipolygonWKT.values[0])

        

        # geojson version:

        polygonsList[cType] = []

        for filename in cType_to_Filename[cType]:

            fullpath = '/'.join([inDir, geojsonDir, sceneId, filename + '.geojson'])

            if not os.path.isfile(fullpath):

                continue

            with open(fullpath, 'r') as fp:

                # print('DEBUG Opening file {}'.format(fullpath))

                fcObj = json.load(fp)

                fcList = fcObj['features']

                for fc in fcList:

                    geom = fc['geometry']

                    polygonsList[cType].append(shapely.geometry.shape(geom))

    return polygonsList
# The main function for this kernel

def get_stats_polygons(polygonsList, image_area=1):

    '''

    Get stats from polygonsList

    '''

    count = {}

    totalArea = {}

    meanArea = {}

    stdArea = {}

    for cType in polygonsList:

        count[cType] = len(polygonsList[cType])

        if count[cType] > 0:

            totalArea[cType] = np.sum ([polygon.area for polygon in polygonsList[cType]]) / image_area*100

            meanArea [cType] = np.mean([polygon.area for polygon in polygonsList[cType]]) / image_area*100

            stdArea  [cType] = np.std ([polygon.area for polygon in polygonsList[cType]]) / image_area*100

        else:

            totalArea[cType] = 0

            meanArea [cType] = np.nan

            stdArea  [cType] = np.nan



    return pd.DataFrame({

            'CLASS' : CLASSES,

            'counts' : count,

            'totalAreas' : totalArea,

            'meanAreas' : meanArea,

            'stdAreas' : stdArea,

            })
def collect_stats():

    pStatsList = []

    for sceneId in trainSceneIds:

        polyList = get_polygons(sceneId)

        pStats = get_stats_polygons(polyList, image_area=get_grid_area(sceneId))

        pStats['SceneId'] = sceneId

        pStatsList.append(pStats)

        print('Stats loaded for ', sceneId)

        # print(pStats)

    return pd.concat(pStatsList)

pStats = collect_stats()

pStats
# Function to pivot and plot

def plot_stats(pStats, values, title):

    pvt = pStats.pivot(index='CLASS', columns='SceneId', values=values)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.set_aspect('equal')

    plt.imshow(pvt, interpolation='nearest', cmap=plt.cm.plasma, extent=[0, 25, 10, 0])

    plt.xticks(np.arange(0.5, 25.4, 1))

    plt.yticks(np.arange(0.5, 10.4, 1))

    ax.set_xticklabels(np.arange(1,26,1))

    ax.set_yticklabels(pvt.index)

    plt.xlabel('Image')

    plt.ylabel('Class Type')

    plt.title(title)

    plt.colorbar()
plot_stats(pStats, 'counts', 'Number of polygons in image by type')
# Now, let's plot percentage of area covered by each type

plot_stats(pStats, 'totalAreas', 'Percentage of area in image by type')

# Since count is high but percentage is low for tree-type,

# we expect its' meanArea per polygon is low.

# Let's verify this.

plot_stats(pStats, 'meanAreas', 'Average area covered by single polygon')
pvt = pStats.pivot(index='CLASS', columns='SceneId', values='totalAreas')

pairImages = ['6110_1_2', '6140_1_2']

print(pvt[pairImages])

from scipy.stats import pearsonr

print('Coverage for different classses in {} and {}: {:5.4f}'.format(

    pairImages[0], pairImages[1], pearsonr(pvt[pairImages[0]],pvt[pairImages[1]])[0]))

# Todo find more pairs
percAreaCS = np.cumsum(pvt, axis=0)

import seaborn as sns



#Set general plot properties

sns.set_style("white")

sns.set_context({"figure.figsize": (12, 8)})



for i in range(1,11):

    cTypeName = percAreaCS.index[-i]

    cTypeId = CLASSES_R[cTypeName]

    ax = sns.barplot(x = percAreaCS.columns, y = percAreaCS.iloc[-i],

                              color = COLORS[cTypeId], label = cTypeName)

l = plt.legend(loc=2)



sns.despine(left=True)

ax.set_xlabel("sceneId")

ax.set_ylabel("%age Area Covered")    

_ = ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=-60)
