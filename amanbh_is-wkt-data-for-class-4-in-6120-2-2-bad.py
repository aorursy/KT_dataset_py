import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import shapely

from shapely.wkt import loads as wkt_loads

# from descartes.patch import PolygonPatch

from matplotlib.patches import Polygon as mPolygon

from matplotlib.patches import Patch as mPatch

import matplotlib.pyplot as plt

import pylab
df = pd.read_csv('../input/train_wkt_v3.csv')

wkt_6120_2_2_4 = df[df.ImageId == '6120_2_2'].iloc[4-1, 2]

sMultiPolygon = wkt_loads(wkt_6120_2_2_4)

print('{} multiploygons in class 4'.format(len(sMultiPolygon)))
badPoly = sMultiPolygon[22]

# Dont care about others



def plot_bad_poly():

    extPoly = mPolygon(badPoly.exterior)

    fig, ax = plt.subplots(figsize=(8,8))

    ax.add_patch(extPoly)

    _ = ax.set_xlim([0, 0.009188])

    _ = ax.set_ylim([-0.0090400000000000012, 0])

    for i in range(len(badPoly.interiors)):

        intPoly = mPolygon(badPoly.interiors[i], color='red', alpha = 0.8, lw=0, ec=None)

        ax.add_patch(intPoly)

    return ax

        

ax = plot_bad_poly()
# Let's make a polygon inside the big rectangle

from shapely.geometry import Polygon as sPolygon

ext = [ (0.0055, -0.0032), (0.0066, -0.0028), (0.0056, -0.0004), (0.0049, -0.0005)]

testPolygon = sPolygon(ext)

print('Is testPolygon completely inside badPoly: ', badPoly.contains(testPolygon))
# Visualize verify that testPolygon is inside badPoly

ax = plot_bad_poly()

_ = ax.add_patch(mPolygon(testPolygon.exterior, color='green', alpha=0.9))
# We can test with another polygon that touches red areas

# sanity-check

ext2 = [ (0.004, -0.0025), (0.007, -0.0025), (0.007, -0.001), (0.004, -0.001)]

testPolygon2 = sPolygon(ext2)

ax = plot_bad_poly()

_ = ax.add_patch(mPolygon(testPolygon2.exterior, color='green', alpha=0.9))

print('Is testPolygon completely inside badPoly: ', badPoly.contains(testPolygon2))