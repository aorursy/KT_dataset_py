import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
NYC=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
NYC.head()
NYC.shape
NYC.info()
NYC.duplicated().sum()
NYC.drop_duplicates(inplace=True)
NYC.isnull().sum()
NYC.drop(['name','id','host_name','last_review'], axis=1, inplace=True)
NYC.tail()
NYC.isnull().sum()
NYC.dropna(how='any',inplace=True)
NYC.info()
NYC.fillna({'reviews_per_month':0}, inplace=True)
NYC.reviews_per_month.isnull().sum()

NYC.describe()
NYC.columns
plt.figure(figsize=(14,6))
sns.heatmap(NYC.corr(),annot=True)
plt.title("Correlation of the following data");

neighbourhoodC=NYC.neighbourhood_group.unique()
sns.countplot(neighbourhoodC)

fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('NEIGHBOURHOOD GROUPS')
#Restaurants delivering Online or not
sns.countplot(NYC['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')
plt.figure(figsize=(10,10))
ax = sns.barplot(data=NYC, x='neighbourhood_group',y='availability_365',palette='plasma')
plt.xlabel('NEIGHBOURHOOD GROUPS')
plt.ylabel('Availability 365 days')
plt.figure(figsize=(10,6))
sns.scatterplot(NYC.longitude,NYC.latitude,hue=NYC.neighbourhood_group)
plt.ioff()
plt.figure(figsize=(10,6))
sns.scatterplot(NYC.longitude,NYC.latitude,hue=NYC.availability_365)
plt.ioff()
plt.figure(figsize=(10,6))
sns.scatterplot(NYC.longitude,NYC.latitude,hue=NYC.room_type)
plt.ioff()
NYC.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)
#examing the changes
NYC.head(5)
NYC.room_type.unique()


    
for column in NYC.columns[NYC.columns.isin(['neighbourhood_group','room_type'])]:
        NYC[column] = NYC[column].factorize()[0]
NYC.head()
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
NYC.columns
X=NYC[['neighbourhood_group', 'room_type','minimum_nights',
       'calculated_host_listings_count', 'availability_365']]
y=NYC['price']
print(X)
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=34)

X_train.shape
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(min_samples_leaf=.0001)
model.fit(X_train, y_train)
model.score(X_train,y_train)
NN_model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'),
                                      tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),
                                      tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),
                                      tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),
                                      tf.keras.layers.Dense(1, kernel_initializer='normal',activation='linear')])





NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
NN_model.fit(X, y, epochs=500, batch_size=32, validation_split = 0.2) 
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
XGBModel = XGBRegressor()
XGBModel.fit(X_train,y_train , verbose=False)


XGBpredictions = XGBModel.predict(X_test)
MAE = mean_absolute_error(y_test , XGBpredictions)
print('XGBoost validation MAE = ',MAE)

