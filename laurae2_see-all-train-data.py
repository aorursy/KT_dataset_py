import pandas as pd

import numpy as np

from shapely.wkt import loads

from matplotlib.patches import Polygon

import matplotlib.pyplot as plt



df = pd.read_csv('../input/train_wkt.csv')
for ids in ['6100_1_3', '6010_4_2', '6010_4_4', '6140_3_1', '6170_2_4', '6040_1_3', '6040_2_2', '6170_4_1', '6110_4_0', '6120_2_2', '6100_2_3', '6120_2_0', '6150_2_3', '6110_1_2', '6170_0_4', '6160_2_1', '6090_2_0', '6140_1_2', '6060_2_3', '6110_3_1', '6040_1_0']:

    polygonsList = {}

    image = df[df.ImageId == ids]

    for cType in image.ClassType.unique():

        polygonsList[cType] = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])

    fig, ax = plt.subplots(figsize=(9, 9))

    for p in polygonsList:

        for polygon in polygonsList[p]:

            mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

            ax.add_patch(mpl_poly)

    ax.relim()

    ax.autoscale_view()

    plt.title(ids)