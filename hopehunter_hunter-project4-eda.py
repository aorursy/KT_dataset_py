# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import xlrd
import contextily as ctx
import geoplot as gplt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read and assign name to training data for EDA
train = pd.read_csv('../input/cee-498-project-4-no2-prediction/train.csv')

#Shape of data frame
print(train.shape)
#show full data frame
pd.set_option("display.max_rows", None, "display.max_columns", None)
#check out the columns of data frame
train.head()
train.dtypes
#examine summary stats of data frame
train.describe()
#Are there missing values in the dataset? - no
print(sum(train.isnull().sum()))
#Are there duplicated in the dataset? - no
print(sum(train.duplicated()))
mean = train['Observed_NO2_ppb'].mean()
std = train['Observed_NO2_ppb'].std()
num_bins = 20

plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(train['Observed_NO2_ppb'], num_bins, density=True)
y = scipy.stats.norm.pdf(bins, mean, std)
plt.plot(bins, y, 'r--')
plt.xlabel('Observed NO2 (ppb)')
plt.ylabel('Probability density')
plt.title('PDF of Observed NO2 (ppb)')
plt.show()


sns.boxplot(x=train["Observed_NO2_ppb"])
States = train["State"].unique()
print(train["Monitor_ID"].nunique())
print(States)
for i in States:
    sub = train[train["State"] == i]
    j = sub["Monitor_ID"].nunique()
    print(i,j)
fig,ax = plt.subplots(1,1)
states = gpd.read_file('../input/usshapefile/tl_2017_us_state.shp')
states.plot(ax=ax, color = 'black')
train_points = train.apply(lambda row: Point(row.Longitude, row.Latitude), axis = 1)
gpd_train = gpd.GeoDataFrame(train, geometry = df_points)
gpd_train.plot(ax=ax, alpha=0.9,legend=True,markersize=10)
plt.xlim(-130,-65)
plt.ylim(24,50)
plt.figure(figsize=(20,15))
plt.show()
fig,ax = plt.subplots(1,1)
states = gpd.read_file('../input/usshapefile/tl_2017_us_state.shp')
states.plot(ax=ax, color = 'grey')
train_points = train.apply(lambda row: Point(row.Longitude, row.Latitude), axis = 1)
gpd_train = gpd.GeoDataFrame(train, geometry = df_points)
gpd_train.plot(column='Observed_NO2_ppb', ax=ax, alpha=0.9,legend=True, 
               markersize=15, legend_kwds={'label': 'Observed NO2 (ppb)', 'orientation':'horizontal'})
plt.xlim(-130,-65)
plt.ylim(24,50)
plt.figure(figsize=(20,15))
plt.show()
print(plt.scatter(train["Observed_NO2_ppb"], train["Impervious_100"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Impervious_1000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Impervious_3000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Impervious_10000"]))
print('Impervious_100:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_100"]))
print('Impervious_200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_200"]))
print('Impervious_300:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_300"]))
print('Impervious_400:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_400"]))
print('Impervious_500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_500"]))
print('Impervious_600:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_600"]))
print('Impervious_700:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_700"]))
print('Impervious_800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_800"]))
print('Impervious_1000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_1000"]))
print('Impervious_1200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_1200"]))
print('Impervious_1500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_1500"]))
print('Impervious_1800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_1800"]))
print('Impervious_2000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_2000"]))
print('Impervious_2500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_2500"]))
print('Impervious_3000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_3000"]))
print('Impervious_3500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_3500"]))
print('Impervious_4000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_4000"]))
print('Impervious_5000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_5000"]))
print('Impervious_6000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_6000"]))
print('Impervious_7000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_7000"]))
print('Impervious_8000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_8000"]))
print('Impervious_10000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Impervious_10000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Population_100"]))
print(plt.scatter(train["Observed_NO2_ppb"], np.log(train["Population_1000"])))
print(plt.scatter(train["Observed_NO2_ppb"], train["Population_3000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Population_10000"]))
train100 = train[train["Population_100"]!= 0]
print(plt.scatter(train2["Observed_NO2_ppb"], np.log(train2["Population_100"])))
print('Population_100:',train100["Observed_NO2_ppb"]. corr(np.log(train2["Population_100"])))
train200 = train[train["Population_200"]!= 0]
train300 = train[train["Population_300"]!= 0]
train400 = train[train["Population_400"]!= 0]
train500 = train[train["Population_500"]!= 0]
train600 = train[train["Population_600"]!= 0]
train700 = train[train["Population_700"]!= 0]
train800 = train[train["Population_800"]!= 0]
train1000 = train[train["Population_1000"]!= 0]
train1200 = train[train["Population_1200"]!= 0]
train1500 = train[train["Population_1500"]!= 0]
train1800 = train[train["Population_1800"]!= 0]
train2000 = train[train["Population_2000"]!= 0]
train2500 = train[train["Population_2500"]!= 0]
train3000 = train[train["Population_3000"]!= 0]
train3500 = train[train["Population_3500"]!= 0]
train4000 = train[train["Population_4000"]!= 0]
train5000 = train[train["Population_5000"]!= 0]
train6000 = train[train["Population_6000"]!= 0]
train7000 = train[train["Population_7000"]!= 0]
train8000 = train[train["Population_8000"]!= 0]
train10000 = train[train["Population_10000"]!= 0]
print(plt.scatter(train2["Observed_NO2_ppb"], np.log(train2["Population_1000"])))
print(plt.scatter(train2["Observed_NO2_ppb"], np.log(train2["Population_3000"])))
print(plt.scatter(train2["Observed_NO2_ppb"], np.log(train2["Population_10000"])))
print('Population_100:',train100["Observed_NO2_ppb"]. corr(np.log(train100["Population_100"])))
print('Population_200:',train200["Observed_NO2_ppb"]. corr(np.log(train200["Population_200"])))
print('Population_300:',train300["Observed_NO2_ppb"]. corr(np.log(train300["Population_300"])))
print('Population_400:',train400["Observed_NO2_ppb"]. corr(np.log(train400["Population_400"])))
print('Population_500:',train500["Observed_NO2_ppb"]. corr(np.log(train500["Population_500"])))
print('Population_600:',train600["Observed_NO2_ppb"]. corr(np.log(train600["Population_600"])))
print('Population_700:',train700["Observed_NO2_ppb"]. corr(np.log(train700["Population_700"])))
print('Population_800:',train800["Observed_NO2_ppb"]. corr(np.log(train800["Population_800"])))
print('Population_1000:',train1000["Observed_NO2_ppb"]. corr(np.log(train1000["Population_1000"])))
print('Population_1200:',train1200["Observed_NO2_ppb"]. corr(np.log(train1200["Population_1200"])))
print('Population_1500:',train1500["Observed_NO2_ppb"]. corr(np.log(train1500["Population_1500"])))
print('Population_1800:',train1800["Observed_NO2_ppb"]. corr(np.log(train1800["Population_1800"])))
print('Population_2000:',train2000["Observed_NO2_ppb"]. corr(np.log(train2000["Population_2000"])))
print('Population_2500:',train2500["Observed_NO2_ppb"]. corr(np.log(train2500["Population_2500"])))
print('Population_3000:',train3000["Observed_NO2_ppb"]. corr(np.log(train3000["Population_3000"])))
print('Population_3500:',train3500["Observed_NO2_ppb"]. corr(np.log(train3500["Population_3500"])))
print('Population_4000:',train4000["Observed_NO2_ppb"]. corr(np.log(train4000["Population_4000"])))
print('Population_5000:',train5000["Observed_NO2_ppb"]. corr(np.log(train5000["Population_5000"])))
print('Population_6000:',train6000["Observed_NO2_ppb"]. corr(np.log(train6000["Population_6000"])))
print('Population_7000:',train7000["Observed_NO2_ppb"]. corr(np.log(train7000["Population_7000"])))
print('Population_8000:',train8000["Observed_NO2_ppb"]. corr(np.log(train8000["Population_8000"])))
print('Population_10000:',train10000["Observed_NO2_ppb"]. corr(np.log(train10000["Population_10000"])))
print(plt.scatter(train["Observed_NO2_ppb"], train["Major_100"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Major_1000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Major_3000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Major_10000"]))
print('Major_100:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_100"]))
print('Major_200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_200"]))
print('Major_300:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_300"]))
print('Major_400:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_400"]))
print('Major_500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_500"]))
print('Major_600:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_600"]))
print('Major_700:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_700"]))
print('Major_800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_800"]))
print('Major_1000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_1000"]))
print('Major_1200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_1200"]))
print('Major_1500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_1500"]))
print('Major_1800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_1800"]))
print('Major_2000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_2000"]))
print('Major_2500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_2500"]))
print('Major_3000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_3000"]))
print('Major_3500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_3500"]))
print('Major_4000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_4000"]))
print('Major_5000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_5000"]))
print('Major_6000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_6000"]))
print('Major_7000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_7000"]))
print('Major_8000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_8000"]))
print('Major_10000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Major_10000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_100"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_1000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_3000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_10000"]))
print('Resident_100:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_100"]))
print('Resident_200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_200"]))
print('Resident_300:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_300"]))
print('Resident_400:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_400"]))
print('Resident_500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_500"]))
print('Resident_600:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_600"]))
print('Resident_700:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_700"]))
print('Resident_800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_800"]))
print('Resident_1000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_1000"]))
print('Resident_1200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_1200"]))
print('Resident_1500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_1500"]))
print('Resident_1800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_1800"]))
print('Resident_2000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_2000"]))
print('Resident_2500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_2500"]))
print('Resident_3000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_3000"]))
print('Resident_3500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_3500"]))
print('Resident_4000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_4000"]))
print('Resident_5000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_5000"]))
print('Resident_6000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_6000"]))
print('Resident_7000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_7000"]))
print('Resident_8000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_8000"]))
print('Resident_10000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["Resident_10000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["total_100"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_1000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_3000"]))
print(plt.scatter(train["Observed_NO2_ppb"], train["Resident_10000"]))
print('total_100:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_100"]))
print('total_200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_200"]))
print('total_300:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_300"]))
print('total_400:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_400"]))
print('total_500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_500"]))
print('total_600:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_600"]))
print('total_700:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_700"]))
print('total_800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_800"]))
print('total_1000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_1000"]))
print('total_1200:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_1200"]))
print('total_1500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_1500"]))
print('total_1800:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_1800"]))
print('total_2000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_2000"]))
print('total_2500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_2500"]))
print('total_3000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_3000"]))
print('total_3500:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_3500"]))
print('total_4000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_4000"]))
print('total_5000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_5000"]))
print('total_6000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_6000"]))
print('total_7000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_7000"]))
print('total_8000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_8000"]))
print('total_10000:', scipy.stats.spearmanr(train["Observed_NO2_ppb"], train["total_10000"]))
plt.scatter(train["Observed_NO2_ppb"], train["WRF+DOMINO"])
plt.scatter(train["Observed_NO2_ppb"], train["Distance_to_coast_km"])
plt.scatter(train["Observed_NO2_ppb"], train["Elevation_truncated_km"])
print("WRF+DOMINO",train["Observed_NO2_ppb"]. corr(train["WRF+DOMINO"]))
print("Distance_to_coast_km",train["Observed_NO2_ppb"]. corr(train["Distance_to_coast_km"])) 
print("Elevation_truncated_km",train["Observed_NO2_ppb"]. corr(train["Elevation_truncated_km"]))