# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#These are libraries for visualization 

import seaborn as sns

import matplotlib.pyplot as plt 



#Setting instances

sns.set()

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/auto-mpg.csv',index_col='car name', )
data.head()
data.shape 
data.dtypes
data.isnull().sum()
data.shape
data.duplicated().sum()
data.cylinders.unique()
data['model year'].unique()
data.origin.unique()
data.horsepower.unique()
data = data[data.horsepower != '?']
#Validting th changes in horsepower column

print('?' in data.horsepower)
#Shape of data  after changes...

data.shape
data.dtypes
data.horsepower = data.horsepower.astype('float')

data.dtypes


data['PW_ratio']= (data.horsepower / data.weight)
data['DispCC']=data['displacement']* 16.3871
data['DispLitr']= data['DispCC']/1000
data.shape
data.describe()
sns.distplot(data['mpg']);

plt.title('MPG Distribution in Data')

plt.show()
sns.distplot(data['acceleration'], hist=True, kde=False, color='red')

plt.title('Acceleration Distribution in Data')

plt.show()
plt.figure(figsize=(12,4))

sns.boxplot(x='cylinders',y='displacement', data=data)

plt.show()
data['displacement'].value_counts().plot(kind='hist');

plt.xlabel('Displacement in Cu Inches')

plt.title('Displacement Distribution')

plt.show()
data['DispCC'].value_counts().plot(kind='hist', color='Green')

plt.xlabel('Displacement CC')

plt.title('Displacement in CC Distribution')

plt.show()
plt.figure(figsize=(10,5))

data['acceleration'].value_counts().head(20).plot(kind='hist', title='Acceleraion of top 20 Cars');
plt.figure(figsize=(14,5))

sns.scatterplot(x='DispLitr', y='mpg', data=data)

plt.title('Mpg against Displacement in Litres wrt Countries')

plt.show()
sns.boxplot(x='cylinders',y='mpg',data=data);

plt.title('Relation between Cylinders and MPG')

plt.show()
sns.boxplot(x='model year',y='mpg',data=data);

plt.title('Relation between Model Year and MPG')

plt.show()
sns.scatterplot(x='PW_ratio', y='mpg', data=data)

plt.title('Power to Weight ratio comparison with miles per Gallon')

plt.show()
sns.scatterplot(x='horsepower', y='mpg', data=data);

plt.title('Relation between Cylinders, Horsepower and MPG')

plt.show()
sns.scatterplot(x='acceleration', y='mpg', data=data);

plt.title('Relation between Acceleration and MPG')

plt.show()
sns.scatterplot(x='displacement', y='mpg',size='weight', hue='weight', data=data)

plt.title('Relation between Acceleration, Weight and MPG')

plt.show()
sns.pairplot(data, height = 2.0,hue ='origin')

plt.title('Comparison between car by manufacturing Countries')

plt.show()
cor = data.corr()

plt.figure(figsize=(12,6))

sns.heatmap(cor, annot=True)

plt.title('Correlation in Auto-MPG Data')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score, mean_squared_error



from sklearn.ensemble import RandomForestRegressor
X = data.drop('mpg', axis=1)

Y = data[['mpg']]

print(X.shape, Y.shape)
scaler = MinMaxScaler()

scaler.fit(X)

X_ = scaler.transform(X)

X = pd.DataFrame(data=X_, columns = X.columns)

X.head()
xtrain,xtest, ytrain, ytest = train_test_split(X, Y,test_size = 0.3, random_state=30, shuffle= True)

print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)
rf_rgr = RandomForestRegressor(criterion='mse', max_depth=5, random_state=30)
rf_rgr.fit(xtrain, ytrain)
rf_rgr.predict(xtest)
print(ytest.head(), rf_rgr.predict(xtest)[0:5])
r2_score(ytest, rf_rgr.predict(xtest))
mean_squared_error(ytest, rf_rgr.predict(xtest))
features_tuple=list(zip(X.columns,rf_rgr.feature_importances_))
feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])

feature_imp=feature_imp.sort_values("Importance",ascending=False)
plt.figure(figsize=(12,4))

sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='g')

plt.xlabel("Auto MPG Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.title("Random Forest Classifier - Features Importance")
from sklearn.model_selection import GridSearchCV
param_grid1 = {"n_estimators" : [9, 18, 27, 36, 45, 54, 63, 72, 81, 90],

           "max_depth" : [1, 5, 10, 15, 20, 25, 30],

           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}



RF = RandomForestRegressor(random_state=30)

# Instantiate the GridSearchCV object: logreg_cv

RF_cv1 = GridSearchCV(RF, param_grid1, cv=5,scoring='r2',n_jobs=4)



# Fit it to the data

RF_cv1.fit(xtrain,ytrain)



#RF_cv1.cv_results_, 

RF_cv1.best_params_, RF_cv1.best_score_
param_grid2 = {"n_estimators" : [72,75,78,81,84,87,90],

           "max_depth" : [5,6,7,8,9,10,11,12,13,14,15],

           "min_samples_leaf" : [1,2,3,4]}



RF = RandomForestRegressor(random_state=30)

# Instantiate the GridSearchCV object: logreg_cv

RF_cv2 = GridSearchCV(RF, param_grid2, cv=5,scoring='r2',n_jobs=4)



# Fit it to the data

RF_cv2.fit(xtrain,ytrain)



#RF_cv2.grid_scores_, 

RF_cv2.best_params_, RF_cv2.best_score_
RF_tuned = RF_cv2.best_estimator_
RF_tuned.fit(xtrain, ytrain)
pred = RF_tuned.predict(xtest)
print(ytest.head(), pred[0:5])
r2_score(ytest, pred)
mean_squared_error(ytest, pred)