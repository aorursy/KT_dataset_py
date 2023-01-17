import pandas as pd

from sklearn import model_selection 
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.isnull().sum()
df.shape
df.describe()
df.Seller_Type.unique()
df.Fuel_Type.unique()
df.Transmission.unique()
df.Owner.unique()
df.Year.unique()
df.drop(['Car_Name'],axis=1,inplace=True)
df.head()
df['current year']=2020
df.head()
df['total_year'] = df['current year'] - df['Year']
df.drop(['Year','current year'],axis=1,inplace=True)
df.head()
df = df.sample(frac=1).reset_index(drop=True) 
df.head()
df = pd.get_dummies(df,drop_first=True)
import seaborn as sns

import matplotlib.pyplot as plt

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X=df.drop(['Selling_Price'],axis=1)

y=df['Selling_Price']
y.shape
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# we are tuning three hyperparameters right now, we are passing the different values for both parameters

grid_param = {

    "n_estimators" : [120,300,500,800,1200],

    'max_depth' : [5, 8, 15, 25, 30],

    'max_features' : ['auto','log2', 'sqrt'],

    'min_samples_split' : [1, 2, 5, 10, 15, 100],

    'min_samples_leaf' : [1, 2, 5, 10]

}

rand_reg = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rand_reg, param_grid=grid_param, cv=5, n_jobs = -1, verbose = 3)
grid_search.fit(X_train,y_train)
grid_search.best_params_
rand_reg_new = RandomForestRegressor(

 max_depth= 8,

 max_features = 'auto',

 min_samples_leaf = 1,

 min_samples_split = 2,

 n_estimators = 120,

 random_state = 42)
rand_reg_new.fit(X_train,y_train)
predictions = rand_reg_new.predict(X_test)
from sklearn import metrics

import numpy as np



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import pickle

# open a file, where you ant to store the data

file = open('random_forest_regression_model.pkl', 'wb')



# dump information to that file

pickle.dump(rand_reg_new, file)