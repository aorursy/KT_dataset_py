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
# Importing libraries and resources 

import plotly.graph_objects as go

import seaborn as sns

%matplotlib inline

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

from sklearn.preprocessing import LabelBinarizer

import statsmodels.api as sm

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import BaggingRegressor

from sklearn import ensemble

# Reading the data

data=pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
# Visualizing the data and the dataset dimensions

print(data.shape)

data.head()
# Ocean proximity to dummie



dummies=LabelBinarizer().fit_transform(data['ocean_proximity'])

data=data.join(pd.DataFrame(dummies,columns=["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"]))



data.tail(2)
# Missing data

data.isnull().sum()
# filling missing data



for i in data['ocean_proximity'].unique():

    median=data[data['ocean_proximity']==i]['total_bedrooms'].median()

    data.loc[data['ocean_proximity']==i,'total_bedrooms'] =  data[data['ocean_proximity']==i]['total_bedrooms'].fillna(median)

    

data.isnull().sum()
# Map chart: Visualizing the Median House Value



california_img=mpimg.imread('/kaggle/input/housingprices/california.png')





data.plot(kind="scatter", x="longitude", y="latitude", title='Median House Value',

    s=data['population']/100, label="population",

    c="median_house_value", cmap=plt.get_cmap("jet"),

    colorbar=True, alpha=0.7, figsize=(10,7),

)





plt.ylabel("Latitude", fontsize=10)

plt.xlabel("Longitude", fontsize=10)



plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=1)



plt.legend()

plt.show()
# interactive map chart with plotly 



fig = go.Figure(data=go.Scattergeo(

        lon = data['longitude'],

        lat = data['latitude'],

        text = data['median_house_value'],

        mode = 'markers',

        marker_color = data['median_house_value'],

        ))



fig.update_layout(

        title = 'Median House Value in California',

        geo_scope='usa')

    

fig.show()
# ocean_proximity analysis 



fig=plt.figure(figsize=(17, 4))



plt.subplot(131)

g = sns.countplot(data=data,x="ocean_proximity",palette="Blues",orient="v",dodge=True).set_title('Ocean Proximity Count')



plt.subplot(132)

sns.boxplot( x=data["ocean_proximity"], y=data["median_house_value"], palette="Blues").set_title('Median House Value Boxplot by Ocean Proximity')



plt.tight_layout()

plt.show()

# Histograma de median_house_value

from scipy.stats import iqr

fig = plt.figure(figsize=(20, 6))



bin_width = 2 * iqr(data["median_house_value"]) / len(data)**(1/3)

num_bins = (np.max(data["median_house_value"]) - np.min(data["median_house_value"])) / bin_width



plt.subplot(131)

(sns.distplot(data["median_house_value"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "median_house_value", ylabel = "Density", title = "Median House Value Histogram"));



plt.subplot(132)

sns.boxplot(y=data["median_house_value"], color="skyblue").set_title('Median House Value Boxplot')



plt.tight_layout()

plt.show()

# Descriptive analysis 

data[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']].describe()
# Histogram



fig = plt.figure(figsize=(15, 5))



plt.subplot(131)



(sns.distplot(data["housing_median_age"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "Housing Median Age", ylabel = "Density", title = "Median House Age Histogram"));



plt.subplot(132)



(sns.distplot(data["total_rooms"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "Total Rooms", ylabel = "Density", title = "Total Rooms Histogram"));



plt.subplot(133)



(sns.distplot(data["total_bedrooms"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "Total Bedrooms", ylabel = "Density", title = "Total Bedrooms Histogram"));



plt.tight_layout()

plt.show()



# Boxplot



fig = plt.figure(figsize=(15, 5))



plt.subplot(131)

sns.boxplot(y=data["housing_median_age"], color="skyblue").set_title('Median House Age Boxplot')



plt.subplot(132)

sns.boxplot(y=data["total_rooms"], color="skyblue").set_title('Total Rooms Boxplot')



plt.subplot(133)

sns.boxplot(y=data["total_bedrooms"], color="skyblue").set_title('Total Bedrooms Boxplot')



plt.tight_layout()

plt.show()





# Histogram



fig = plt.figure(figsize=(15, 5))



plt.subplot(131)



(sns.distplot(data["population"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "Population", ylabel = "Density", title = "Population Histogram"));



plt.subplot(132)



(sns.distplot(data["households"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "Households", ylabel = "Density", title = "Households Histogram"));



plt.subplot(133)



(sns.distplot(data["median_income"], bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "Median Income", ylabel = "Density", title = "Median Income Histogram"));



plt.tight_layout()

plt.show()





# Boxplot 



fig = plt.figure(figsize=(15, 5))



plt.subplot(131)

sns.boxplot(y=data["population"], color="skyblue").set_title('Population Boxplot')



plt.subplot(132)

sns.boxplot(y=data["households"], color="skyblue").set_title('Households Boxplot')



plt.subplot(133)

sns.boxplot(y=data["median_income"], color="skyblue").set_title('Median Income Boxplot')



plt.tight_layout()

plt.show()

(data[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']]

 .corr().style.background_gradient( axis=None))
# Removing categorical column  

data=data.drop(['ocean_proximity'],axis=1)

print(data.shape)

data.head()
#  splitting the data into attributes and labels:

#  response variable: meadian_house_value



X = data.drop(['median_house_value'],axis=1).values

y = data.iloc[:, 8].values
# splitting the data into training and testing sets:

# 20% for testing 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# feature scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Linear Regression 



from sklearn.linear_model import LinearRegression



regressor_linear = LinearRegression()



regressor_linear.fit(X_train,y_train)



y_pred_lr = regressor_linear.predict(X_test) 

print('Mean Absolute Error:', np.round(metrics.mean_absolute_error(y_test, y_pred_lr),2))



print('Mean Squared Error:', np.round(metrics.mean_squared_error(y_test, y_pred_lr),2))



print('Root Mean Squared Error:', np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)),2))
# residual plot

plt.figure(figsize=(12,7))

(sns.distplot((y_test-y_pred_lr), bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "(y_test-y_pred)", ylabel = "Density", title = "Regression Tree Residual Plot"));
# training the model

regressor_tree = DecisionTreeRegressor(random_state = 123)  

regressor_tree.fit(X_train, y_train)

y_pred_tree = regressor_tree.predict(X_test) 
# evaluating the algorithm 



print('Mean Absolute Error:', np.round(metrics.mean_absolute_error(y_test, y_pred_tree),2))



print('Mean Squared Error:', np.round(metrics.mean_squared_error(y_test, y_pred_tree),2))



print('Root Mean Squared Error:', np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_tree)),2))
# residual plot

plt.figure(figsize=(12,7))

(sns.distplot((y_test-y_pred_tree), bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "(y_test-y_pred)", ylabel = "Density", title = "Regression Tree Residual Plot"));
# training the model



regressor_bagging = BaggingRegressor(n_estimators=200,random_state=123)

regressor_bagging=regressor_bagging.fit(X_train,y_train)

y_pred_bagging = regressor_bagging.predict(X_test)

# evaluating the algorithm 



print('Mean Absolute Error:', np.round(metrics.mean_absolute_error(y_test, y_pred_bagging),2))



print('Mean Squared Error:', np.round(metrics.mean_squared_error(y_test, y_pred_bagging),2))



print('Root Mean Squared Error:', np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_bagging)),2))



# residual plot

plt.figure(figsize=(12,7))

(sns.distplot((y_test-y_pred_bagging), bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "(y_test-y_pred)", ylabel = "Density", title = "RF Residual Plot"));

# training the model

regressor_rf = RandomForestRegressor(n_estimators=200, random_state=123)

regressor_rf.fit(X_train, y_train)

y_pred_rf = regressor_rf.predict(X_test)
# evaluating the algorithm 



print('Mean Absolute Error:', np.round(metrics.mean_absolute_error(y_test, y_pred_rf),2))



print('Mean Squared Error:', np.round(metrics.mean_squared_error(y_test, y_pred_rf),2))



print('Root Mean Squared Error:', np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)),2))

# residual plot

plt.figure(figsize=(12,7))

(sns.distplot((y_test-y_pred_rf), bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "(y_test-y_pred)", ylabel = "Density", title = "RF Residual Plot"));

# setting the parameters

params = {

    'n_estimators': 200,

    'learning_rate': 0.015,

    'max_depth': 10,

    'min_samples_split': 2,

    'loss': 'ls',

}



# training the model

regressor_boosting = ensemble.GradientBoostingRegressor(**params)

regressor_boosting.fit(X_train, y_train)

y_pred_boosting=regressor_boosting.predict(X_test)
# evaluating the algorithm 



print('Mean Absolute Error:', np.round(metrics.mean_absolute_error(y_test, y_pred_boosting),2))



print('Mean Squared Error:', np.round(metrics.mean_squared_error(y_test, y_pred_boosting),2))



print('Root Mean Squared Error:', np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_boosting)),2))



# residual plot

plt.figure(figsize=(12,7))

(sns.distplot((y_test-y_pred_boosting), bins = "fd", norm_hist = True, kde = False, color = "skyblue", hist_kws = dict(alpha = 1))

    .set(xlabel = "(y_test-y_pred)", ylabel = "Density", title = "Boosting Residual Plot"));


