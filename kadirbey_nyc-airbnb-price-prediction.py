# importing modules which are going to use



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from math import sqrt

from sklearn.metrics import r2_score
# the first look through the data 



NYC_AIRBNB = pd.read_csv('../input/AB_NYC_2019.csv', encoding = "ISO-8859-1")

NYC_AIRBNB.head()
# data types and numbers of variables

NYC_AIRBNB.info()
# info of NaN in our data set as percentage



def show_missing (df):

    """This function returns percentage and total number of missing values"""

    percent = df.isnull().sum()*100/df.shape[0]

    total = df.isnull().sum()

    missing = pd.concat([percent, total], axis=1, keys=['percent', 'total'])

    return missing[missing.total>0].sort_values('total', ascending=False)
show_missing(NYC_AIRBNB)
# dropping 'id', host_name', 'last_review' and 'host_id' columns

# Because of the low percentage of "host_name"s NANs, "last_review" is a date. 

# we don't use 'id', and 'host_id' in our model.



NYC_AIRBNB=NYC_AIRBNB.drop(['id','host_name','last_review', 'host_id'], axis=1)
#dropping the NaNs from 'name'



NYC_AIRBNB=NYC_AIRBNB.dropna(subset=["name"])
# filling the NaNs of "reviews_per_month" with mean



NYC_AIRBNB['reviews_per_month'] = NYC_AIRBNB['reviews_per_month'].fillna(NYC_AIRBNB['reviews_per_month'].mean())
NYC_AIRBNB.describe()
NYC_AIRBNB.price.hist(bins=100)
NYC_AIRBNB[NYC_AIRBNB.price<1000].price.hist(bins=100)

plt.show()
from scipy.stats import zscore
z=zscore(NYC_AIRBNB.price)

z
m=NYC_AIRBNB.price.mean()

s=NYC_AIRBNB.price.std()

m+3*s
plt.boxplot(NYC_AIRBNB.price)

plt.show()
# we will drop the rows which have a price more than mean+3*std (zscore>3)

np.sum(NYC_AIRBNB.price>873)
NYC_AIRBNB_dropped=NYC_AIRBNB[NYC_AIRBNB.price<873]

NYC_AIRBNB_dropped
from scipy.stats.mstats import winsorize



NYC_AIRBNB_winsor = NYC_AIRBNB.copy()

NYC_AIRBNB_winsor['price'] = winsorize(NYC_AIRBNB["price"], (0, 0.01))



max(NYC_AIRBNB_winsor.price)
plt.boxplot(NYC_AIRBNB_winsor.price)

plt.title("Boxplot of Prices")

plt.show()
import plotly.graph_objs as go
# create a donut-like pie chart to show the ratio of Airbnb numbers by Neighborhood_group



colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

labels = NYC_AIRBNB.neighbourhood_group.value_counts().index

values = NYC_AIRBNB.neighbourhood_group.value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.2)])

fig.update_traces(marker=dict(colors = colors, line=dict(color='#000000', width=1)))



fig.show()
# use countplot to show the ratio of Airbnb amounts by Neighborhood_group



plt.subplots(figsize=(14,6))

sns.countplot('neighbourhood_group',data=NYC_AIRBNB,palette='Spectral')

plt.xticks(rotation=70)

plt.title('Airbnb Numbers by Neighbourhood Group in NYC', color='red')

plt.show()
# create a donut-like pie chart to show the ratio of Airbnb room types



colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

labels = NYC_AIRBNB.room_type.value_counts().index

values = NYC_AIRBNB.room_type.value_counts().values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_traces(marker=dict(colors = colors, line=dict(color='#000000', width=2)))



fig.show()
# the ratio of Airbnb room types with countplot by Neighbourhood_group



plt.subplots(figsize=(14,6))

sns.countplot('neighbourhood_group',data=NYC_AIRBNB,palette='Spectral', hue="room_type")

plt.xticks(rotation=70)

plt.title('Airbnb Room Types by Neighbourhood Group in NYC', color='red')

plt.show()
# neighbourhood_groups with daily average prices



NYC_AIRBNB.groupby('neighbourhood_group').mean()['price'].plot.bar()
#let's grab 10 most reviewed listings in NYC



top_reviewed_listings=NYC_AIRBNB.nlargest(10,'number_of_reviews')

top_reviewed_listings
price_avrg=top_reviewed_listings.price.mean()

print('Average price per night: {}'.format(price_avrg))
NYC_AIRBNB['Cat'] = NYC_AIRBNB['price'].apply(lambda x: 'costly' if x > 3000

                                                    else ('medium' if x >= 1000 and x < 3000

                                                    else ('reasonable' if x >= 500 and x < 1000

                                                     else ('cheap' if x >= 100 and x <500

                                                          else'very cheap'))))
plt.figure(figsize=(10,8))



sns.scatterplot(NYC_AIRBNB.latitude,NYC_AIRBNB.longitude, hue='Cat', data=NYC_AIRBNB)
data_manhattan=NYC_AIRBNB[NYC_AIRBNB.neighbourhood_group=='Manhattan']

data_manhattan.head()
data_manha_65=data_manhattan[data_manhattan.price<65]

data_manha_65['label']=data_manha_65.apply(lambda x: (x['name'],'price:'+str(x['price'])),axis=1)

data_manha_65.head()
import folium

from folium import plugins
# According to this map,you can see the number of rooms due to price of <65 USD.



Long=-73.92

Lat=40.86

manha_map=folium.Map([Lat,Long],zoom_start=12)



manha_rooms_map=plugins.MarkerCluster().add_to(manha_map)

for lat,lon,label in zip(data_manha_65.latitude,data_manha_65.longitude,data_manha_65.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(manha_rooms_map)

manha_map.add_child(manha_rooms_map)



manha_map
NYC_AIRBNB_model = NYC_AIRBNB_dropped.copy()
# using get_dummies by concatting "room_type","neighbourhood_group" to shape our model 



NYC_AIRBNB_model= pd.concat([NYC_AIRBNB_model, pd.get_dummies(NYC_AIRBNB_model[["room_type","neighbourhood_group"]])], axis=1)

NYC_AIRBNB_model.head()
plt.figure(figsize=(10,10))

sns.distplot(NYC_AIRBNB_model['price'], fit=norm)

plt.title("Price Distribution Plot",size=15, weight='bold')
NYC_AIRBNB_model['price_log'] = np.log(NYC_AIRBNB_model.price+1)
# With Log-Price Distribution Plot we see price feature have normal distribution.



plt.figure(figsize=(12,10))

sns.distplot(NYC_AIRBNB_model['price_log'], fit=norm)

plt.title("Log-Price Distribution Plot",size=12, weight='bold')
plt.figure(figsize=(7,7))

stats.probplot(NYC_AIRBNB_model['price_log'], plot=plt)

plt.show()
# Need to see the correlation table but it shows us that there is no strong relationship between price and other features.

# But at least price_log will give us some correleations with room_types.



f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(NYC_AIRBNB_model.drop('neighbourhood', axis = 1).corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#trying to create our model with _price_log



Y = NYC_AIRBNB_model['price_log']

X = NYC_AIRBNB_model.drop(["neighbourhood", "neighbourhood_group", "room_type", "price", 

                           "latitude", "longitude","price_log", "name"], axis=1)
lrm = LinearRegression()

lrm.fit(X, Y)
print('Variables: \n', lrm.coef_)

print('Constant Value (bias): \n', lrm.intercept_)
import statsmodels.api as sm



X = sm.add_constant(X)

results = sm.OLS(Y, X).fit()

results.summary()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm

from statsmodels.tools.eval_measures import mse, rmse



%matplotlib inline

pd.options.display.float_format = '{:.3f}'.format



import warnings

warnings.filterwarnings(action="ignore")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 465)



print("Observation number in train set : {}".format(X_train.shape[0]))

print("Observation number in test set   : {}".format(X_test.shape[0]))
X_train = sm.add_constant(X_train)



results = sm.OLS(y_train, X_train).fit()



results.summary()
# real values with red line, predicted values with blue dots



X_test = sm.add_constant(X_test)

y_preds = results.predict(X_test)



baslik_font = {'family': 'arial','color':  'darkred','weight': 'bold','size': 15 }

eksen_font = {'family': 'arial','color':  'darkblue','weight': 'bold','size': 10 }

plt.figure(dpi = 100)



plt.scatter(y_test, y_preds)

plt.plot(y_test, y_test, color="red")

plt.xlabel("Real Values", fontdict=eksen_font)

plt.ylabel("Predicted Values", fontdict=eksen_font)

plt.title("Prices: Real and Predicted Values", fontdict=baslik_font)

plt.show()



print("mean_absolute_error (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))

print("mean_squared_error (MSE)          : {}".format(mse(y_test, y_preds)))

print("root_mean_squared_error (RMSE)     : {}".format(rmse(y_test, y_preds)))

print("mean_absolute_percentage_error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 465)
lrm = LinearRegression()

lrm.fit(X_train, y_train)



y_train_prediction = lrm.predict(X_train)

y_test_prediction = lrm.predict(X_test)



print("Train Observation Number  : {}".format(X_train.shape[0]))

print("Test Observation Number    : {}".format(X_test.shape[0]), "\n")



print("R-Square value in Train Set  : {}".format(lrm.score(X_train, y_train)))

print("-----Test Set Statistics---")

print("Test set R-Square Value         : {}".format(lrm.score(X_test, y_test)))

print("mean_absolute_error (MAE)        : {}".format(mean_absolute_error(y_test, y_test_prediction)))

print("mean_squared_error (MSE)          : {}".format(mse(y_test, y_test_prediction)))

print("root_mean_squared_error (RMSE)     : {}".format(rmse(y_test, y_test_prediction)))

print("mean_absolute_percentage_error (MAPE) : {}".format(np.mean(np.abs((y_test - y_test_prediction) / y_test)) * 100))
for alpha in [10**x for x in range (-10, 10)]:

    ridgeregr = Ridge(alpha=alpha) 

    ridgeregr.fit(X_train, y_train)



    y_train_prediction = ridgeregr.predict(X_train)

    y_test_prediction = ridgeregr.predict(X_test)



    print("alpha : {} , root_mean_squared_error (RMSE) : {}".format(alpha, rmse(y_test, y_test_prediction)))
ridgeregr = Ridge(alpha=1) 

ridgeregr.fit(X_train, y_train)



y_train_prediction = ridgeregr.predict(X_train)

y_test_prediction = ridgeregr.predict(X_test)



print("R-Square value in Train Set       : {}".format(ridgeregr.score(X_train, y_train)))

print("-----Test Set Statistics---")

print("Test set R-Square Value         : {}".format(ridgeregr.score(X_test, y_test)))

print("mean_absolute_error (MAE)        : {}".format(mean_absolute_error(y_test, y_test_prediction)))

print("mean_squared_error (MSE)          : {}".format(mse(y_test, y_test_prediction)))

print("root_mean_squared_error (RMSE)     : {}".format(rmse(y_test, y_test_prediction)))
mse(y_test, y_test_prediction)
from sklearn.linear_model import Lasso



lassoregr = Lasso(alpha=0.0001) 

lassoregr.fit(X_train, y_train)



y_train_prediction = lassoregr.predict(X_train)

y_test_prediction = lassoregr.predict(X_test)



print("R-Square value in Train Set       : {}".format(lassoregr.score(X_train, y_train)))

print("-----Test Set Statistics---")

print("Test set R-Square Value         : {}".format(lassoregr.score(X_test, y_test)))

print("mean_absolute_error (MAE)        : {}".format(mean_absolute_error(y_test, y_test_prediction)))

print("mean_squared_error (MSE)          : {}".format(mse(y_test, y_test_prediction)))

print("root_mean_squared_error (RMSE)     : {}".format(rmse(y_test, y_test_prediction)))
for alpha in [10**x for x in range (-10, 10)]:

    lassoregr = Lasso(alpha=alpha) 

    lassoregr.fit(X_train, y_train)



    y_train_prediction = lassoregr.predict(X_train)

    y_test_prediction = lassoregr.predict(X_test)



    print("alpha : {} , root_mean_squared_error (RMSE) : {}".format(alpha, rmse(y_test, y_test_prediction)))
from sklearn.linear_model import ElasticNet



elasticregr = ElasticNet(alpha=0.001, l1_ratio=0.5) 

elasticregr.fit(X_train, y_train)



y_train_prediction = elasticregr.predict(X_train)

y_test_prediction = elasticregr.predict(X_test)



print("R-Square value in Train Set       : {}".format(elasticregr.score(X_train, y_train)))

print("-----Test Set Statistics---")

print("Test set R-Square Value          : {}".format(elasticregr.score(X_test, y_test)))

print("mean_absolute_error (MAE)        : {}".format(mean_absolute_error(y_test, y_test_prediction)))

print("mean_squared_error (MSE)          : {}".format(mse(y_test, y_test_prediction)))

print("root_mean_squared_error (RMSE)     : {}".format(rmse(y_test, y_test_prediction)))
np.exp (y_test_prediction)