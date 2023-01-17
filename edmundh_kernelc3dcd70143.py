# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import cartopy.crs as ccrs

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/real-estate-price-prediction/Real estate.csv", index_col="No")
df.tail()
hpp = sns.pairplot(df, corner=True)  # corner: get rid of duplicate "noise" from upper triangle
hms = plt.matshow(df.corr(), cmap="RdBu_r", vmin=-1, vmax=1)

hcb = plt.colorbar(hms)
hjp = sns.jointplot(x="X6 longitude", y="X5 latitude", data=df)
varibs = [

    "Y house price of unit area",

    "X3 distance to the nearest MRT station",

    "X4 number of convenience stores",

    "X2 house age",

    "X1 transaction date",

]



fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=[10,8],

                         subplot_kw={'projection': ccrs.PlateCarree()})



# Set background color for fig and subplots to sth to help see data

# over full range without changing away from default viridis cmap

back_col = "darkgrey"

fig.patch.set_facecolor(back_col)



# Name lats and lons for convenience

lons = df["X6 longitude"]

lats = df["X5 latitude"]



# Get the (tight) bounds

x0, x1 = lons.min(), lons.max()

y0, y1 = lats.min(), lats.max()

m = 0.01  # Add a margin - set_xmargin / plt.margins seem to be ignored :(



vaxes = axes.flatten()[:-1]  # 5 variables, but 6 axes - make our zip simpler by lopping off!

axes[-1,-1].set_visible(False)  # Hide unused final axis https://stackoverflow.com/a/10035974

for varib, vax in zip(varibs, vaxes):

    plt.sca(vax)

    vax.background_patch.set_facecolor(back_col)  # https://github.com/SciTools/cartopy/issues/880

    vax.set_extent([x0-m, x1+m, y0-m, y1+m], ccrs.PlateCarree())

    

    hs = plt.scatter(lons, lats, c=df[varib], transform=ccrs.PlateCarree(), s=1)

    plt.colorbar(hs)

    

    plt.title(varib)