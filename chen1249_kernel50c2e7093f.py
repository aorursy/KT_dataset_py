import numpy as np                

import pandas as pd 

import matplotlib            

import matplotlib.pyplot as plt

import plotly as plotly  

import sklearn

from sklearn import preprocessing

from sklearn import metrics

import seaborn as sns 

import seaborn as sb

import matplotlib.pyplot as plt

sb.set() 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston

from sklearn.metrics import mean_absolute_error
data=pd.read_csv('../input/seattle/listings.csv')

data.head()
data.price = data['price'].str.replace('$', '')

data.price= pd.to_numeric(data['price'], errors='coerce')

data['price'].isnull().sum()
data= data.dropna(subset=['price'])

data['price'].isnull().sum()
data.neighbourhood_group_cleansed.isnull().sum()
data.room_type.isnull().sum()
data.review_scores_rating.isnull().sum()
data= data.dropna(subset=['review_scores_rating'])

data['price'].isnull().sum()
data.price = np.log(data['price'])
import plotly.offline as pyo

import plotly.graph_objs as go

roomdf = data.groupby('room_type').size()/data['room_type'].count()*100

labels = roomdf.index

values = roomdf.values



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
plt.figure(figsize=(20,10))

sb.distplot(data[data.room_type=='Entire home/apt'].price,color='maroon',hist=False,label='Entire home/apt')

sb.distplot(data[data.room_type=='Private room'].price,color='black',hist=False,label='Private room')

sb.distplot(data[data.room_type=='Shared room'].price,color='green',hist=False,label='Shared room')

plt.title('room_type and price')

plt.xlim(0,10)

plt.show()
plt.figure(figsize=(40,20))

sb.boxplot(x="price",y ='room_type' ,data = data)

plt.title("room_type price distribution")

plt.xticks(rotation='horizontal')

plt.show()
#catplot room type and price

plt.figure(figsize=(20,6))

sns.catplot(x='room_type', y='price', data=data);

plt.ioff()
f,axes = plt.subplots(1,1,figsize=(24,12))

sb.boxplot(data=data['price'],orient="h")
relate = pd.DataFrame(data[['price','review_scores_rating']])

relate.corr()
sb.pairplot(data=relate)
roomdf = data.groupby('neighbourhood_group_cleansed').size()/data['neighbourhood_group_cleansed'].count()*100

labels = roomdf.index

values = roomdf.values



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
plt.figure(figsize=(20,15))

sb.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group_cleansed)

plt.ioff()
import folium

import folium.plugins as plugins

from folium.plugins import HeatMap

m=folium.Map(location = [47.65,-122.30],zoom_start = 12)

hm = HeatMap(data=data[['latitude','longitude']].dropna(),radius=12,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'})

hm.add_to(m)

display(m)
plt.figure(figsize=(10,16))

sb.boxplot(y="price",x ='neighbourhood_group_cleansed' ,data = data)

plt.title("neighbourhood_group price distribution")

plt.show()
data1 = pd.DataFrame({'location': data['neighbourhood_group_cleansed'],'roomtype':data['room_type'],'score_rating':data['review_scores_rating'],'price':data['price']})

data1.head()
data1.info()
le = preprocessing.LabelEncoder()                                            # Fit label encoder

le.fit(data['neighbourhood_group_cleansed'])

feature1=le.transform(data['neighbourhood_group_cleansed'])    # Transform labels to normalized encoding.

le.fit(data['room_type'])

feature2=le.transform(data['room_type'])

df = pd.DataFrame({'location': feature1,'roomtype':feature2,'score_rating':data['review_scores_rating'],'price':data['price']})

df.head()
# Import essential models and functions from sklearn

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# Extract Response and Predictors

predictors = ["location", "roomtype","score_rating"]



y = pd.DataFrame(df["price"])

X = pd.DataFrame(df[predictors])



# Split the Dataset into Train and Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



# Linear Regression using Train Data

linreg = LinearRegression()         # create the linear regression object

linreg.fit(X_train, y_train)        # train the linear regression model



# Coefficients of the Linear Regression line

print('Intercept of Regression \t: b = ', linreg.intercept_)

print('Coefficients of Regression \t: a = ', linreg.coef_)

print()



# Print the Coefficients against Predictors

print(pd.DataFrame(list(zip(X_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"]))

print()



# Predict Response corresponding to Predictors

y_train_pred = linreg.predict(X_train)

y_test_pred = linreg.predict(X_test)



# Plot the Predictions vs the True values

f, axes = plt.subplots(1, 2, figsize=(24, 12))

axes[0].set_ylim([0, 10])

axes[1].set_ylim([0, 10])

axes[0].scatter(y_train, y_train_pred, color = "blue")

axes[0].plot(y_train, y_train, 'b-', linewidth = 1)

axes[0].set_xlabel("True values of the Response Variable (Train)")

axes[0].set_ylabel("Predicted values of the Response Variable (Train)")

axes[1].scatter(y_test, y_test_pred, color = "green")

axes[1].plot(y_test, y_test, 'b-', linewidth = 1)

axes[1].set_xlabel("True values of the Response Variable (Test)")

axes[1].set_ylabel("Predicted values of the Response Variable (Test)")

plt.show()



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))

print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))

print()



# Check the Goodness of Fit (on Test Data)

print("Goodness of Fit of Model \tTest Dataset")

print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))

print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))

print()
data_dummies = pd.get_dummies(data[['neighbourhood_group_cleansed','room_type','review_scores_rating','price']])

data_dummies.head()
predictors = ['neighbourhood_group_cleansed_Ballard',

 'neighbourhood_group_cleansed_Beacon Hill',

 'neighbourhood_group_cleansed_Capitol Hill',

 'neighbourhood_group_cleansed_Cascade',

 'neighbourhood_group_cleansed_Central Area',

 'neighbourhood_group_cleansed_Delridge',

 'neighbourhood_group_cleansed_Downtown',

 'neighbourhood_group_cleansed_Interbay',

 'neighbourhood_group_cleansed_Lake City',

 'neighbourhood_group_cleansed_Magnolia',

 'neighbourhood_group_cleansed_Northgate',

 'neighbourhood_group_cleansed_Other neighborhoods',

 'neighbourhood_group_cleansed_Queen Anne',

 'neighbourhood_group_cleansed_Rainier Valley',

 'neighbourhood_group_cleansed_Seward Park',

 'neighbourhood_group_cleansed_University District',

 'neighbourhood_group_cleansed_West Seattle', "room_type_Entire home/apt",'room_type_Private room','room_type_Shared room',"review_scores_rating"]



y = pd.DataFrame(data_dummies["price"])

X = pd.DataFrame(data_dummies[predictors])



# Split the Dataset into Train and Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



# Linear Regression using Train Data

linreg = LinearRegression()         # create the linear regression object

linreg.fit(X_train, y_train)        # train the linear regression model



# Coefficients of the Linear Regression line

print('Intercept of Regression \t: b = ', linreg.intercept_)

print('Coefficients of Regression \t: a = ', linreg.coef_)

print()



# Print the Coefficients against Predictors

print(pd.DataFrame(list(zip(X_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"]))

print()



# Predict Response corresponding to Predictors

y_train_pred = linreg.predict(X_train)

y_test_pred = linreg.predict(X_test)



# Plot the Predictions vs the True values

f, axes = plt.subplots(1, 2, figsize=(24, 12))

axes[0].set_ylim([0, 10])

axes[1].set_ylim([0, 10])

axes[0].scatter(y_train, y_train_pred, color = "blue")

axes[0].plot(y_train, y_train, 'b-', linewidth = 1)

axes[0].set_xlabel("True values of the Response Variable (Train)")

axes[0].set_ylabel("Predicted values of the Response Variable (Train)")

axes[1].scatter(y_test, y_test_pred, color = "green")

axes[1].plot(y_test, y_test, 'b-', linewidth = 1)

axes[1].set_xlabel("True values of the Response Variable (Test)")

axes[1].set_ylabel("Predicted values of the Response Variable (Test)")

plt.show()



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))

print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))

print()



# Check the Goodness of Fit (on Test Data)

print("Goodness of Fit of Model \tTest Dataset")

print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))

print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))

print()