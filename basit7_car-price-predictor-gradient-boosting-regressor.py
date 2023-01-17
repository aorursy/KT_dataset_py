# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
#import datafile
df = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")
df.shape
df.head()
# Check for unique values
print(df["seller_type"].unique())
print(df["fuel"].unique())
print(df["transmission"].unique())
print(df["owner"].unique())
print(df["seller_type"].value_counts())
print(df["fuel"].value_counts())
print(df["transmission"].value_counts())
print(df["owner"].value_counts())
#Removing unneccessary data
final_dataset = df.drop("name", axis = 1)
final_dataset.drop(final_dataset[final_dataset['fuel']=='Electric'].index,axis=0,inplace=True)
final_dataset.drop(final_dataset[final_dataset['owner']=='Test Drive Car'].index,axis=0,inplace=True)
final_dataset.shape
#Adding a new column car_Age & removing year column
final_dataset["car_Age"] = 2020 - final_dataset["year"]
final_dataset.drop(["year"],axis = 1, inplace=True)
final_dataset['no_of_previous_owners'] = final_dataset['owner'].map({'First Owner':1,'Second Owner':2,'Third Owner':3,"Fourth & Above":4})
final_dataset = final_dataset.dropna()
final_dataset['no_of_previous_owners'] = final_dataset['no_of_previous_owners'].astype(int)
final_dataset.drop(['owner'], axis=1, inplace=True)
final_dataset.head()
final_dataset.drop(final_dataset[final_dataset['seller_type']=='Trustmark Dealer'].index,axis=0,inplace=True)
#Encoding the categorical data
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset.shape
final_dataset.head()
# Looking for Correlation
#Dependence of target with every variable (ranging from -1 to 1)
corr_matrix = final_dataset.corr()
corr_matrix['selling_price'].sort_values(ascending=False)
#checking the correlation between variables
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
#Independent and dependent variable(s)
X = final_dataset.drop("selling_price", axis=1)      #final_dataset.iloc[:,1:]
y = final_dataset[["selling_price"]]                 #final_dataset.iloc[:,0]
#feature importance

model = ExtraTreesRegressor()
model.fit(X,y)
#looking for most important features
#plotting the graph of feature importance for better visualisation
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(6).plot(kind='bar')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
print("Length of traing data set is :{} and length of test set is :{}".format(len(X_train), len(X_test)))
#Randomized search CV for random forest

#Number of treers in random forest
n_estimators =[int(x) for x in np.linspace(100,1200,num = 12)]
#Number of features to consider at every split
max_features =["auto", "sqrt"]
#Maximum number of levels in the tree 
max_depth = [int(x) for x in np.linspace(5, 30,6)]
#Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
#Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
#create the random forest grid
rf_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf}
print(rf_grid)
#Randomized search CV for gradient boosting
#Number of treers in random forest
n_estimators =[int(x) for x in np.linspace(100,1200,num = 12)]
#Learning rate
learning_rate = [0.01, 0.02, 0.05, 0.1, 0.2]
subsample = [0.05, 0.06, 0.08, 0.09, 0.1]
criterion = ['mse', 'rmse', 'friedman_mse']
#Number of features to consider at every split
max_features =["auto", "sqrt"]
#creating gradient boosting grid
gb_grid = {'n_estimators' : n_estimators,
           'learning_rate' : learning_rate,
           'subsample' : subsample,
           'max_depth' : max_depth,
           'max_features' : max_features}
print(gb_grid)
#Use the random grid to search the best parameters
#Create the base model to tune
rf_model = RandomForestRegressor()
final_rf_model = RandomizedSearchCV(estimator = rf_model, 
                                 param_distributions=rf_grid,
                                 scoring='neg_mean_squared_error',
                                 n_iter = 20,
                                 cv = 5,
                                 verbose = 2,
                                 random_state = 42,
                                 n_jobs =1)
final_rf_model.fit(X_train,y_train)
final_rf_model.best_params_
y_pred = final_rf_model.predict(X_test)
sns.distplot(y_test['selling_price']-y_pred)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor()
final_gb_model = RandomizedSearchCV(estimator = gb_model, 
                                 param_distributions=gb_grid,
                                 scoring='neg_mean_squared_error',
                                 n_iter = 20,
                                 cv = 5,
                                 verbose = 2,
                                 random_state = 42,
                                 n_jobs =1)
final_gb_model.fit(X_train, y_train)
final_gb_model.best_params_
y_pred = final_gb_model.predict(X_test)
sns.distplot(y_test['selling_price']-y_pred)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
