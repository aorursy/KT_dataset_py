import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import shapely

from shapely.wkt import loads as wkt_loads
# Read the training data from WKT format

inDir = '../input'

df = pd.read_csv(inDir + '/train_wkt_v4.csv',

        names=['SceneId', 'ClassType', 'MultipolygonWKT'], skiprows=1)

trainSceneIds = np.sort(df.SceneId.unique())

classes = range(1,11)



# Function to load polygons

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

        None if sceneId is missing from df

    '''

    df_scene = df[df.SceneId == sceneId]

    if len(df_scene) == 0:

        return None

    polygonsList = {}

    for cType in classes:

        polygonsList[cType] = wkt_loads(df_scene[df_scene.ClassType == cType].MultipolygonWKT.values[0])

    return polygonsList
# Locate invalid polygons in the training data

for sceneId in trainSceneIds:

    pl = get_polygons(sceneId)

    for cType in classes:

        if not pl[cType].is_valid:

            # One of the polygons in this MultiPolygon is invalid

            for i, poly in enumerate(pl[cType]):

                if not poly.is_valid:

                    print('Scene {} Class {} Polygon {} is invalid'.format(sceneId, cType, i))

                    fixed_poly = poly.buffer(0)  # Fix invalid polygon

                    print('Polygon fixed? :', fixed_poly.is_valid)                   