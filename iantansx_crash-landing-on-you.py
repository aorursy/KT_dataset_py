# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from statsmodels.graphics.mosaicplot import mosaic


from mpl_toolkits.basemap import Basemap
from matplotlib import cm

from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import math
import re
import datetime
import pandas as pd
df = pd.read_csv("../input/aviation-accident-database-synopses/AviationData.csv",encoding = "ISO-8859-1")
df.head()
# splitting date field in the components

df['Year'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").year)
df['Month'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").month)
df['Day'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").day)

df_timeseries = df[df['Year'] >= 1982]

# For the time series charts I start sorting data
df_timeseries = df_timeseries.sort_values(by=['Year', 'Month', 'Day'], ascending=True)

years = np.arange(1982, 2017)

sns.set(style="darkgrid")

plt.subplot(211)

g = sns.countplot(x="Year", data=df_timeseries, palette="GnBu_d", order=years)
g.set_xticklabels(labels=years)
a = plt.setp(g.get_xticklabels(), rotation=90)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15, 10))
fig.subplots_adjust(hspace=.6)
colors = ['#99cc33', '#a333cc', '#333dcc']
df['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,0], kind='bar', title='Phase of Flight')
df['Broad.Phase.of.Flight'].value_counts().plot(ax=axes[0,1], kind='pie', title='Phase of Flight')
df['Weather.Condition'].value_counts().plot(ax=axes[1,0], kind='pie', colors=colors, title='Weather Condition')
# TODO: clean up to add "other"
# ds['cleaned.make'].value_counts().plot(ax=axes[1,1], kind='pie', title='Aircraft Make')
#cleaning the predcitors 
df['Make'] = df["Make"].str.lower()
df['Engine.Type'].fillna('None',inplace = True)
df['Weather.Condition'].fillna('unknown',inplace = True)
#cleaning outcome y 
df.loc[(df['Injury.Severity'] != "Non-Fatal") & (df['Injury.Severity'] != "Incident"), 'Injury.Severity'] = 'Fatal'
df.loc[(df['Injury.Severity'] == "Incident"), 'Injury.Severity'] = 'Fatal'
df['Injury.Severity'].value_counts()
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
#splitting the data
predictors = ['Weather.Condition','Engine.Type','Make']
one_hot_data = pd.get_dummies(df[predictors],drop_first=True)
# Train Set : 1100 samples
df_train = pd.DataFrame(df[:67409])
hot_predictor_train = pd.DataFrame(one_hot_data[:67409])
y_train = pd.DataFrame(df_train['Injury.Severity'])
X_train = hot_predictor_train 

# Test Set : 360 samples
df_test = pd.DataFrame(df[-16853:])
hot_predictor_test = pd.DataFrame(one_hot_data[-16853:])
y_test = pd.DataFrame(df_test['Injury.Severity'])
X_test = hot_predictor_test
# Decision Tree using Train Data
dectree = DecisionTreeClassifier(max_depth = 3)  # create the decision tree object
dectree.fit(X_train, y_train)  
def dectree_pred(X_train,X_test) :
    # Predict Response corresponding to Predictors
    y_train_pred = dectree.predict(X_train)
    y_test_pred = dectree.predict(X_test)

    # Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Classification Accuracy \t:", dectree.score(X_train, y_train))
    print()

    listall_train = confusion_matrix(y_train, y_train_pred)
    listtop_train = listall_train[0]
    listbot_train = listall_train[1]
    fpr_train= listtop_train[1]/(sum(listtop_train))
    fnr_train = listbot_train[0]/(sum(listbot_train))
    print('The False Positive Rate is \t:{0:2f}'.format(fpr_train))
    print('The False Negative Rate is \t:{0:2f}'.format(fnr_train))
    print()

    # Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Classification Accuracy \t:", dectree.score(X_test, y_test))
    print()

    listall_test = confusion_matrix(y_test, y_test_pred)
    listtop_test = listall_test[0]
    listbot_test = listall_test[1]
    fpr_test= listtop_test[1]/(sum(listtop_test))
    fnr_test = listbot_test[0]/(sum(listbot_test))
    print('The False Positive Rate is \t:{0:2f}'.format(fpr_test))
    print('The False Negative Rate is \t:{0:2f}'.format(fnr_test))


    # Plot the Confusion Matrix for Train and Test
    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_train, y_train_pred),
               annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])
    sns.heatmap(confusion_matrix(y_test, y_test_pred), 
               annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])

#1 is substantial , 0 is Fatal

dectree_pred(X_train,X_test)
# Export the Decision Tree as a dot object
treedot = export_graphviz(dectree,                                      # the model
                          feature_names = X_test.columns,          # the features 
                          out_file = None,                              # output file
                          filled = True,                                # node colors
                          rounded = True,                               # make pretty
                          special_characters = True)                    # postscript

# Render using graphviz
import graphviz
graphviz.Source(treedot)
df['Aircraft.Damage'].fillna('unknown',inplace = True)
df.loc[(df['Aircraft.Damage'] != "Destroyed") , 'Aircraft.Damage'] = 'Substantial'
df['Aircraft.Damage'].value_counts()
y_train = pd.DataFrame(df_train['Aircraft.Damage'])
y_test = pd.DataFrame(df_test['Aircraft.Damage'])
# Decision Tree using Train Data
dectree = DecisionTreeClassifier(max_depth = 3)  # create the decision tree object
dectree.fit(X_train, y_train)                    # train the decision tree model
dectree_pred(X_train,X_test)
# Export the Decision Tree as a dot object
treedot = export_graphviz(dectree,                                      # the model
                          feature_names = X_test.columns,          # the features 
                          out_file = None,                              # output file
                          filled = True,                                # node colors
                          rounded = True,                               # make pretty
                          special_characters = True)                    # postscript

# Render using graphviz
import graphviz
graphviz.Source(treedot)
from scipy import stats
def cramers_corrected_stat(confusion_matrix, correction: bool) -> float:
    """Calculate the Cramer's V corrected stat for two variables.

    Args:
        confusion_matrix: Crosstab between two variables.
        correction: Should the correction be applied?

    Returns:
        The Cramer's V corrected stat for the two variables.
    """
    chi2 = stats.chi2_contingency(confusion_matrix, correction=correction)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Deal with NaNs later on
    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
        rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
        kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
        corr = np.sqrt(phi2corr / min((kcorr - 1.0), (rcorr - 1.0)))
    return corr
df['Location'] = df["Location"].str.upper() #making all CAPS
df['Location'].fillna('unknown',inplace = True) #removing nan
#removing locations with frequency less than 100 for faster computing 
col = 'Location'  # 'bar'
n = 100  # 2
df_filtered  = df[df.groupby(col)[col].transform('count').ge(n)]
#droping columns that do not hold any true value to prediction or clustering
list_to_drop = ['Event.Id','Investigation.Type','Accident.Number','Event.Date','Country','Report.Status','Publication.Date']
df_filtered = df_filtered.drop(list_to_drop, axis=1)
#creating a confusion matrix 
df_cramer = pd.DataFrame()
count = 0 
for n in df_filtered:
    for i in df_filtered:
        confusion_matrix = pd.crosstab(df_filtered[n], df_filtered[i]).values
        value = cramers_corrected_stat(confusion_matrix,True)
        df_cramer.loc[n,i] = value
        count += 1 
f, axes = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(df_cramer, vmin=0, vmax=1, square = True)
f, axes = plt.subplots(figsize=(20, 25))
ax  = mosaic(df_filtered, ['Location', 'Injury.Severity'], title='DataFrame as Source', gap = 0.001 ,ax= axes ,horizontal = False)
plt.show()
f, axes = plt.subplots(figsize=(20, 25))
ax  = mosaic(df_filtered, ['Location', 'Aircraft.Damage'], title='DataFrame as Source', gap = 0.001 ,ax= axes , axes_label = True , horizontal = False)
plt.show()
#Selecting variables for clustering 
df_cluster = df[['Location','Injury.Severity','Aircraft.Damage','Make','Amateur.Built','Engine.Type']]
#removing NAN
df_cluster = df_cluster.replace(np.nan, 'Unknown', regex=True)
from kmodes.kmodes import KModes
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(df_cluster)
fitClusters_cao
clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = df_cluster.columns
clusterCentroidsDf
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(df_cluster)
fitClusters_huang
cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(df_cluster)
    cost.append(kmode.cost_)
y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(df_cluster)
fitClusters_cao
clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([df_cluster, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index'], axis = 1)
cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]
f, axes = plt.subplots(6,2 ,figsize=(20, 10))

sns.countplot(x ='Injury.Severity' , data =cluster_0 , ax = axes[0,0] )
sns.countplot(x ='Aircraft.Damage' , data =cluster_0 , ax = axes[1,0] )
sns.countplot(x ='Make' , data =cluster_0 , ax = axes[2,0] ,order = pd.value_counts(cluster_0['Make']).iloc[:5].index)
sns.countplot(x ='Amateur.Built' , data =cluster_0 , ax = axes[3,0] )
sns.countplot(x ='Engine.Type' , data =cluster_0 , ax = axes[4,0],order = pd.value_counts(cluster_0['Engine.Type']).iloc[:5].index)
sns.countplot(x ='Location' , data =cluster_0 , ax = axes[5,0],order = pd.value_counts(cluster_0['Location']).iloc[:5].index)


sns.countplot(x ='Injury.Severity' , data =cluster_1 , ax = axes[0,1] )
sns.countplot(x ='Aircraft.Damage' , data =cluster_1 , ax = axes[1,1] )
sns.countplot(x ='Make' , data =cluster_1 , ax = axes[2,1] ,order = pd.value_counts(cluster_1['Make']).iloc[:5].index)
sns.countplot(x ='Amateur.Built' , data =cluster_1 , ax = axes[3,1] )
sns.countplot(x ='Engine.Type' , data =cluster_1 , ax = axes[4,1],order = pd.value_counts(cluster_1['Engine.Type']).iloc[:5].index)
sns.countplot(x ='Location' , data =cluster_1 , ax = axes[5,1],order = pd.value_counts(cluster_1['Location']).iloc[:5].index)

df['cluster_predicted'] = combinedDf['cluster_predicted']

df_location_data = df[['Location','Latitude','Longitude','cluster_predicted']]
df_location_data = df_location_data.dropna()

cluster0_locs =  df_location_data[df_location_data['cluster_predicted'] == 0]
cluster1_locs =  df_location_data[df_location_data['cluster_predicted'] == 1]

from mpl_toolkits.basemap import Basemap
from matplotlib import cm
centroid_region0 = cluster0_locs.loc[cluster0_locs['Location'] == clusterCentroidsDf.at[0, 'Location']]
centroid_region1 = cluster1_locs.loc[cluster1_locs['Location'] == clusterCentroidsDf.at[1, 'Location']]
fig = plt.figure()
plt.figure(figsize=(15,15))

m = Basemap(
    llcrnrlon=-165,
    llcrnrlat=20,
    urcrnrlon=-40,
    urcrnrlat=70,
    projection='cyl',
    resolution='c',
    area_thresh=None,
    rsphere=6370997.0,
    no_rot=False,
    suppress_ticks=True,
    satellite_height=35786000,
    boundinglat=None,
    fix_aspect=True,
    anchor='C',
    celestial=False,
    round=False,
    epsg=None,
    ax=None,
)
x, y = m(cluster0_locs['Longitude'].values, cluster0_locs['Latitude'].values)
m.drawcoastlines()
m.drawcountries()
m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)
m.scatter(centroid_region0['Longitude'], centroid_region0['Latitude'], 50, color='g')
fig = plt.figure()
plt.figure(figsize=(15,15))

m = Basemap(
    llcrnrlon=-165,
    llcrnrlat=20,
    urcrnrlon=-40,
    urcrnrlat=70,
    projection='cyl',
    resolution='c',
    area_thresh=None,
    rsphere=6370997.0,
    no_rot=False,
    suppress_ticks=True,
    satellite_height=35786000,
    boundinglat=None,
    fix_aspect=True,
    anchor='C',
    celestial=False,
    round=False,
    epsg=None,
    ax=None,
)
x, y = m(cluster1_locs['Longitude'].values, cluster1_locs['Latitude'].values)
m.drawcoastlines()
m.drawcountries()
m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)
m.scatter(centroid_region1['Longitude'], centroid_region1['Latitude'], 50, color='g')
from sklearn.cluster import KMeans
latlon = df_location_data[['Longitude', 'Latitude']]
latlon.head()
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(latlon)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
kmeans = KMeans(n_clusters=3)
kmodel = kmeans.fit(latlon)
centroids = kmodel.cluster_centers_
centroids
lons, lats = zip(*centroids)
fig = plt.figure()
plt.figure(figsize=(15,15))
north, south, east, west = 71.39, 24.52, -66.95, 172.5
m = Basemap(
    llcrnrlon=-135,
    llcrnrlat=-20,
    urcrnrlon=86,
    urcrnrlat=60,
    projection='cyl',
    resolution='c',
    area_thresh=None,
    rsphere=6370997.0,
    no_rot=False,
    suppress_ticks=True,
    satellite_height=35786000,
    boundinglat=None,
    fix_aspect=True,
    anchor='C',
    celestial=False,
    round=False,
    epsg=None,
    ax=None,
)
x, y = m(df_location_data['Longitude'].values, df_location_data['Latitude'].values)
m.drawcoastlines()
m.drawcountries()
m.hexbin(x, y, gridsize=1000, bins='log', cmap=cm.YlOrRd)
cx, cy = m(lons, lats)
m.scatter(cx, cy, 50, color='g')
