# Importing libraries
import pandas as pd
from pandas import DataFrame 
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import operator
import statistics
import warnings
import seaborn as sns
import plotly.plotly as py
from operator import itemgetter, attrgetter
#Importing data
regression_train = pd.read_csv('../input/train.csv',sep=r'\s*,\s*',engine='python')
regression_test = pd.read_csv('../input/test.csv',sep=r'\s*,\s*',engine='python')
regression_train.shape
regression_train.head()
regression_train.describe()
# Correlation analysisimport plotly.plotly as py
corr = regression_train.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(regression_train.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(regression_train.columns)
ax.set_yticklabels(regression_train.columns)
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
%matplotlib inline
 
west, south, east, north = -125, 32.55, -114, 41.95
 
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
ax.set_title("Housing distribution over the Californian territory")
 
m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
x, y = m(regression_train['longitude'].values, regression_train['latitude'].values)
m.hexbin(x, y, gridsize=100,
         bins='log', cmap=cm.YlOrRd_r);
# Boxplot visualization
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["median_age"])

plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.median_age, 50, density=1)
ax.set_title("median_age distribution")
fig.tight_layout()
plt.show()
# Scatter graph
_, ax = plt.subplots(figsize=(16,8))
ax.set_title("median_age x median_house_value")
ax.axis([0, 60, 0, 500000])
plt.plot(regression_train.median_age, regression_train.median_house_value, 'ro')
plt.show()
# Removing rows in which median_age is equal 52, as there are a full vertical of them w/ this characteristic
train_data1 = regression_train[~(regression_train['median_age'] == 52)]
train_data1.shape
#Boxplot
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["total_rooms"])
plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.total_rooms, 25, density=1)
ax.set_title("total_rooms distribution")
fig.tight_layout()
plt.show()
# Scatter graph
_, ax = plt.subplots(figsize=(16,8))
ax.set_title("total_rooms x median_house_value")
plt.plot(regression_train.total_rooms, regression_train.median_house_value, 'ro')
plt.show()
# removing rows in which total_rooms is equal or bigger than 25,000
train_data2 = train_data1[~(train_data1['total_rooms'] >= 25000)]
train_data2.shape
# Boxplot
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["total_bedrooms"])

plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.total_bedrooms, 25, density=1)
ax.set_title("total_bedrooms distribution")
fig.tight_layout()
plt.show()
# Scatter graph
_, ax = plt.subplots(figsize=(16,8))
ax.set_title("total_bedrooms x median_house_value")
plt.plot(regression_train.total_bedrooms, regression_train.median_house_value, 'ro')
plt.show()
# removing rows in which total_bedrooms is equal or bigger than 3,000
train_data3 = train_data2[~(train_data2['total_bedrooms'] >= 3000)]
train_data3.shape
#Boxplot
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["population"])

plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.population, 50, density=1)
ax.set_title("population distribution")
fig.tight_layout()
plt.show()
# Scatter graph
_, ax = plt.subplots(figsize=(16,8))
ax.set_title("population x median_house_value")
plt.plot(regression_train.population, regression_train.median_house_value, 'ro')
plt.show()
# removing rows in which population is equal or bigger than 10,000
train_data4 = train_data3[~(train_data3['population'] >= 10000)]
train_data4.shape
# Boxplot
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["households"])

plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.households, 25, density=1)
ax.set_title("households distribution")
fig.tight_layout()
plt.show()
# Scatter graph
_, ax = plt.subplots(figsize=(16,8))
ax.set_title("households x median_house_value")
plt.plot(regression_train.households, regression_train.median_house_value, 'ro')
plt.show()
# removing rows in which households is equal or bigger than 4,000
train_data5 = train_data4[~(train_data4['households'] >= 4000)]
train_data5.shape
# Boxplot
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["median_income"])

plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.median_income, 25, density=1)
ax.set_title("median_income distribution")
fig.tight_layout()
plt.show()
# Scatter graph
_, ax = plt.subplots(figsize=(16,8))
ax.set_title("median_income x median_house_value")
plt.plot(regression_train.median_income, regression_train.median_house_value, 'ro')
plt.show()
# removing rows in which median_income is equal or bigger than 140,000
train_data6 = train_data5[~(train_data5['median_income'] >= 140000)]
train_data6.shape
#Boxplot
fig, ax = plt.subplots(figsize=(16,8))
sns.set(style="whitegrid")
ax = sns.boxplot(x=regression_train["median_house_value"])

plt.show()
# Histogram
fig, ax = plt.subplots(figsize=(16,8))
n, bins, patches = plt.hist(regression_train.median_house_value, 50, density=1)
ax.set_title("median_house_value distribution")
fig.tight_layout()
plt.show()
# removing rows in which median_house_value is equal to 500,001
train_data7 = train_data6[~(train_data6['median_house_value'] == 500001)]
train_data7.head()
train_data7.shape
train_data7.index = range(len(train_data7.index))
# Taking X and Y for the regressors
size_train = train_data7.shape
# Testing best columns to use in knn regressor
knn_trainX = train_data7[["longitude", "median_age","total_rooms", "total_bedrooms","households", "population","median_income"]]
# Testing best columns to use in lasso regressor
lasso_trainX = train_data7[["longitude","latitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
# Testing best columns to use in ridge regressor
ridge_trainX = train_data7[["longitude", "latitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
# Label
trainY = train_data7.iloc[:,(size_train[1])-1]
# Metrics
from sklearn.metrics import mean_squared_error as mse
import math
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
from sklearn.neighbors import KNeighborsRegressor
kneighbors = KNeighborsRegressor(n_neighbors=2)
kneighbors.fit(knn_trainX,trainY) 
# Prediction
knn_prediction = kneighbors.predict(knn_trainX)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
# Performance estimate
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
warnings.filterwarnings("ignore")
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.5)
clf.fit(lasso_trainX,trainY)
# Prediction
lasso_predict = clf.predict(lasso_trainX)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
# Performance estimate
print(rmsle(df_l.Y_real,df_l.Y_pred))
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(ridge_trainX,trainY) 
# Prediction
ridge_predict = clf.predict(ridge_trainX)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
# Performance estimate
print(rmsle(df_r.Y_real,df_r.Y_pred))
test_data = regression_test[["longitude", "median_age","total_rooms", "total_bedrooms","households", "population","median_income"]]
knn_predict = kneighbors.predict(test_data)
submission = pd.DataFrame(columns=['Id','median_house_value'])
submission.Id = regression_test.Id
submission.median_house_value = knn_predict
submission
submission.to_csv("submission.csv", index=False)