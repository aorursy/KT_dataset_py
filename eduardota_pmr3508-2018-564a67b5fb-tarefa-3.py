# Bibliotecas
#import os
#print(os.listdir("../input"))
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# Datasets
df = pd.read_csv("../input/atividade-3-pmr3508/train.csv")
dfEvaluation = pd.read_csv("../input/atividade-3-pmr3508/test.csv")
# Head
df.head()
#Drop
df = df.drop('Id', axis =1)
df.head()
#Dividir em duas partes
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:,'longitude':'median_income'], df.median_house_value, random_state=0, test_size=0.25)
train = pd.concat([X_train, Y_train], axis=1, sort=False)
test = pd.concat([X_test, Y_test], axis=1, sort=False)
train.head()
test.head()
train['total_rooms_per_capita'] = train['total_rooms']/train['population']
train['total_bedrooms_per_capita'] = train['total_bedrooms']/train['population']
train['households_per_capita'] = train['households']/train['population']
train['median_income_per_capita'] = train['median_income']/train['population']
train.head()
dist2coast = pd.read_csv("../input/distance-to-coast/dist2coast.txt"
                         , sep='\t',
                        names = ['Longitude', 'Latitude', 'Distance_to_coast'])
dist2coast.head()
dist2coast = dist2coast[dist2coast.Longitude >= -130]
dist2coast = dist2coast[dist2coast.Longitude <= -100]
dist2coast = dist2coast[dist2coast.Latitude >= 30]
dist2coast = dist2coast[dist2coast.Latitude <= 50]
dist2coast.head() 
dist2coast.shape 
#dist2coast.as_matrix(columns = ['Longitude', 'Latitude'])
#dist2coast.Distance_to_coast.values
from scipy.interpolate import griddata as gd
#ip(dist2coast.as_matrix(columns = ['Longitude', 'Latitude']),
#  dist2coast.Distance_to_coast.values, [-129.98, 49.98])
#rbf = interp2d(dist2coast.Longitude.values,
#         dist2coast.Longitude.values,
#         dist2coast.Distance_to_coast.values)
#rbf([-129.98, 49.98])
#[-129.98, 49.98]
train['dist_2_coast'] = gd(dist2coast.as_matrix(columns = ['Longitude', 'Latitude']),
  dist2coast.Distance_to_coast.values, train.as_matrix(columns = ['longitude', 'latitude']),
  method = 'cubic')
train.head()
from geopy.distance import vincenty
silicon_valley = [37.37, -122.04]
train['dist_2_silicon_valley'] = ''
dist_2_silicion_valey_array = []
for i in range(train.shape[0]):
    dist_2_silicion_valey_array.append(vincenty((train.as_matrix(columns = ['latitude', 'longitude'])[i][0],
                       train.as_matrix(columns = ['latitude', 'longitude'])[i][1]),
                      silicon_valley).km)
train['dist_2_silicon_valley'] = dist_2_silicion_valey_array
train.head()
test['total_rooms_per_capita'] = test['total_rooms']/test['population']
test['total_bedrooms_per_capita'] = test['total_bedrooms']/test['population']
test['households_per_capita'] = test['households']/test['population']
test['median_income_per_capita'] = test['median_income']/test['population']
test['dist_2_coast'] = gd(dist2coast.as_matrix(columns = ['Longitude', 'Latitude']),
  dist2coast.Distance_to_coast.values, test.as_matrix(columns = ['longitude', 'latitude']),
  method = 'cubic')
test['dist_2_silicon_valley'] = ''
dist_2_silicion_valey_array = []
for i in range(test.shape[0]):
    dist_2_silicion_valey_array.append(vincenty((test.as_matrix(columns = ['latitude', 'longitude'])[i][0],
                       test.as_matrix(columns = ['latitude', 'longitude'])[i][1]),
                      silicon_valley).km)
test['dist_2_silicon_valley'] = dist_2_silicion_valey_array
test.head()
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
sample = train.sample(100)
lat = sample.latitude.values
lon = sample.longitude.values
value = sample.median_house_value
area = 900-sample.dist_2_silicon_valley

# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=37.5, lon_0=-119,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population
# and size reflecting area
m.scatter(lon, lat, latlon=True,
          c=value, s=area,
          cmap='Reds', alpha=0.7)

# 3. create colorbar and legend
plt.colorbar(label=r'Median House Value')
plt.clim(sample.median_house_value.min(), sample.median_house_value.max())

# make legend with dummy points
for a in [area.min(),
          area.max()/2,
          area.max()]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(900-a) + ' km')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left');
plt.title('Distance to Silicon Valley')
area = 300-sample.dist_2_coast

# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=37.5, lon_0=-119,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population
# and size reflecting area
m.scatter(lon, lat, latlon=True,
          c=value, s=area,
          cmap='Reds', alpha=0.7)

# 3. create colorbar and legend
plt.colorbar(label=r'Median House Value')
plt.clim(value.min(), value.max())

# make legend with dummy points
for a in [area.min(), area.max()/2, area.max()]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(300-a) + ' km')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left');
plt.title('Distance to Nearest Coast')
sample = train.sample(1000)
lat = sample.latitude.values
lon = sample.longitude.values
value = sample.median_house_value
area = 10

# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=37.5, lon_0=-119,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population
# and size reflecting area
m.scatter(lon, lat, latlon=True,
          c=value, s=area,
          cmap='Reds', alpha=0.7)

# 3. create colorbar and legend
plt.colorbar(label=r'Median House Value')
plt.clim(value.min(), value.max())

# make legend with dummy points
#https://www.latlong.net/place/los-angeles-ca-usa-1531.html
los_angeles = [34.052235, -118.243683]

train['dist_2_los_angeles'] = ''
dist_2_los_angeles_array = []
for i in range(train.shape[0]):
    dist_2_los_angeles_array.append(vincenty((train.as_matrix(columns = ['latitude', 'longitude'])[i][0],
                       train.as_matrix(columns = ['latitude', 'longitude'])[i][1]),
                      los_angeles).km)
train['dist_2_los_angeles'] = dist_2_los_angeles_array
train.head()
test['dist_2_los_angeles'] = ''
dist_2_los_angeles_array = []
for i in range(test.shape[0]):
    dist_2_los_angeles_array.append(vincenty((test.as_matrix(columns = ['latitude', 'longitude'])[i][0],
                       test.as_matrix(columns = ['latitude', 'longitude'])[i][1]),
                      los_angeles).km)
test['dist_2_los_angeles'] = dist_2_los_angeles_array
test.head()
train["min_dist_2_centre"] = train[["dist_2_silicon_valley", "dist_2_los_angeles"]].min(axis = 1)
test["min_dist_2_centre"] = test[["dist_2_silicon_valley", "dist_2_los_angeles"]].min(axis = 1)
from sklearn.model_selection import cross_val_score as cvs
from sklearn.neighbors import KNeighborsRegressor as knr

features = ["longitude", "latitude", "median_age",
           "total_rooms", "total_bedrooms", "population",
            "households", "median_income",
            "total_rooms_per_capita",
            "total_bedrooms_per_capita",
            "households_per_capita",
           "median_income_per_capita", "dist_2_coast",
            "min_dist_2_centre"]
Xtrain = train[features]
Ytrain = train["median_house_value"]

knn_scores = []
for i in range(1,100,5):
    knn = knr(n_neighbors = i)
    scores = cvs(knn, Xtrain, Ytrain, scoring='neg_mean_squared_error',
                cv = 5)
    knn_scores.append([i, -scores.mean()])
knn_scores = np.array(knn_scores)
knn_scores
plt.plot(knn_scores[:,0], knn_scores[:,1])
knn_scores[np.where(knn_scores[:,1] == np.amin(knn_scores[:,1]))[0]]
from sklearn.linear_model import Ridge
features = ["median_age", "total_rooms",
            "total_bedrooms", "population",
            "households", "median_income",
            "total_rooms_per_capita",
            "total_bedrooms_per_capita",
            "households_per_capita",
           "median_income_per_capita", "dist_2_coast",
            "min_dist_2_centre"]
Xtrain = train[features]
Ytrain = train["median_house_value"]

r = Ridge(alpha=1.0)
scores = cvs(r, Xtrain, Ytrain,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
from sklearn.linear_model import Lasso
features = ["median_age", "total_rooms",
            "total_bedrooms", "population",
            "households", "median_income",
            "total_rooms_per_capita",
            "total_bedrooms_per_capita",
            "households_per_capita",
           "median_income_per_capita", "dist_2_coast",
            "min_dist_2_centre"]
Xtrain = train[features]
Ytrain = train["median_house_value"]

l = Lasso(alpha=1.0)
scores = cvs(l, Xtrain, Ytrain,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
x = np.arange(3)
scores = [6.41585939e+09, 4490715905.575688, 4490929445.739623]

fig, ax = plt.subplots()
plt.bar(x, scores)
plt.xticks(x, ('Knn', 'Ridge', 'Lasso'))
plt.title("RMSE")
plt.show()
r = Ridge(alpha=1.0)
features = ["median_age", "total_rooms",
            "total_bedrooms", "population",
            "households", "median_income",
            "total_rooms_per_capita",
            "total_bedrooms_per_capita",
            "households_per_capita",
           "median_income_per_capita", "dist_2_coast",
            "min_dist_2_centre"]

Xtrain = train[features]
Ytrain = train["median_house_value"]
Xtest = test[features]
Ytest = test["median_house_value"]

r.fit(Xtrain, Ytrain)
YPredict = r.predict(Xtest)
from sklearn.metrics import mean_squared_log_error
YPredict = pd.Series(YPredict)
#Temos predições negativas, elas terão que ser truncadas
#o modelo deixa de ser linear
YPredict.describe()
for i in range(YPredict.shape[0]-1):
    if(YPredict[i] < 0):
        YPredict[i] = 0
YPredict.describe()
from math import log, sqrt
Sum = 0
for i in range(YPredict.shape[0]-1):
    Sum +=  (log(1+YPredict[i]) - log(1+np.array(Ytest)[i]))*(log(1+YPredict[i]) - log(1+np.array(Ytest)[i]))
score = sqrt(Sum/YPredict.shape[0])
print(score)
#dfEvaluation
dfEvaluation['total_rooms_per_capita'] = dfEvaluation['total_rooms']/dfEvaluation['population']
dfEvaluation['total_bedrooms_per_capita'] = dfEvaluation['total_bedrooms']/dfEvaluation['population']
dfEvaluation['households_per_capita'] = dfEvaluation['households']/dfEvaluation['population']
dfEvaluation['median_income_per_capita'] = dfEvaluation['median_income']/dfEvaluation['population']
dfEvaluation['dist_2_coast'] = gd(dist2coast.as_matrix(columns = ['Longitude', 'Latitude']),
  dist2coast.Distance_to_coast.values, dfEvaluation.as_matrix(columns = ['longitude', 'latitude']),
  method = 'cubic')
dfEvaluation['dist_2_silicon_valley'] = ''
dist_2_silicion_valey_array = []
for i in range(dfEvaluation.shape[0]):
    dist_2_silicion_valey_array.append(vincenty((dfEvaluation.as_matrix(columns = ['latitude', 'longitude'])[i][0],
                       dfEvaluation.as_matrix(columns = ['latitude', 'longitude'])[i][1]),
                      silicon_valley).km)
dfEvaluation['dist_2_silicon_valley'] = dist_2_silicion_valey_array
dfEvaluation['dist_2_los_angeles'] = ''
dist_2_los_angeles_array = []
for i in range(dfEvaluation.shape[0]):
    dist_2_los_angeles_array.append(vincenty((dfEvaluation.as_matrix(columns = ['latitude', 'longitude'])[i][0],
                       dfEvaluation.as_matrix(columns = ['latitude', 'longitude'])[i][1]),
                      los_angeles).km)
dfEvaluation['dist_2_los_angeles'] = dist_2_los_angeles_array
dfEvaluation["min_dist_2_centre"] = dfEvaluation[["dist_2_silicon_valley", "dist_2_los_angeles"]].min(axis = 1)
features = ["median_age", "total_rooms",
            "total_bedrooms", "population",
            "households", "median_income",
            "total_rooms_per_capita",
            "total_bedrooms_per_capita",
            "households_per_capita",
           "median_income_per_capita", "dist_2_coast",
            "min_dist_2_centre"]
XEval = dfEvaluation[features]
YEval = r.predict(XEval)

for i in range(YEval.shape[0]-1):
    if(YEval[i] < 0):
        YEval[i] = 0
Evaluation = pd.DataFrame(columns = ["Id", "median_house_value"])
Evaluation["Id"] = dfEvaluation["Id"]
Evaluation["median_house_value"] = YEval
Evaluation.to_csv("Evaluation.csv")
Evaluation
