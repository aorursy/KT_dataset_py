%matplotlib inline

import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as pl
from matplotlib import animation
import seaborn as sns
def newPalettes():
    palList = []
    colors = [sns.xkcd_rgb["light red"], "red","crimson","orange", "yellow", sns.xkcd_rgb["bright green"],
              "green", sns.xkcd_rgb["forest green"],"cyan","teal","navy","fuchsia","purple"]
    for c in colors:
        palList.append(sns.light_palette(c, as_cmap=True))

    return palList

palettes = newPalettes()
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]


z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))
#train = train[1:300000]

#Get rid of the bad lat/longs
train['Xok'] = train[train.X < -121].X
train['Yok'] = train[train.Y < 40].Y
train["Dates"] = pd.to_datetime(train["Dates"], errors='raise')
train = train.dropna()
trainP = train[train.Category == 'PROSTITUTION']

trainList = []
for i in range(2003, 2016):
    trainList.append(trainP[trainP.Dates.dt.year == i])
pl.figure(figsize=(20, 20 * asp))
for index, trainL in enumerate(trainList):
    pal = palettes[index]
    ax = pl.hexbin(trainL.Xok, trainL.Yok, cmap=pal,
                  bins=5,
                  mincnt=1)
    ax = sns.kdeplot(trainL.Xok, trainL.Yok, clip=clipsize,
                     cmap=pal,
                     aspect=(1 / asp))
    
ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
pl.draw()
categoryList = ["WARRANTS", "ASSAULT", "RUNAWAY"]
for cat in categoryList:
    trainC = train[train.Category == cat]
    trainList = []
    for i in range(2003, 2016):
        trainList.append(trainC[trainC.Dates.dt.year == i])

    pl.figure(figsize=(20, 20 * asp))
    for index, trainL in enumerate(trainList):
        pal = palettes[index]
        ax = pl.hexbin(trainL.Xok, trainL.Yok, cmap=pal,
                      bins=5,
                      mincnt=1)
        ax = sns.kdeplot(trainL.Xok, trainL.Yok, clip=clipsize,
                         cmap=pal,
                         aspect=(1 / asp))

    pl.title(cat)
    ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
                  extent=lon_lat_box,
                  aspect=asp)
    pl.draw()
trainM = train[train.Category == "DRUNKENNESS"]
monthlyList = []
for i in range(1,13):
    monthlyList.append(trainM[trainM.Dates.dt.month == i])
pl.figure(figsize=(20, 20 * asp))
for index, mon in enumerate(monthlyList):
    pal = palettes[index]
    ax = pl.hexbin(mon.Xok, mon.Yok, cmap=pal,
                  bins=5,
                  mincnt=1)
    ax = sns.kdeplot(mon.Xok, mon.Yok, clip=clipsize,
                     cmap=pal,
                     aspect=(1 / asp))

ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
pl.draw()