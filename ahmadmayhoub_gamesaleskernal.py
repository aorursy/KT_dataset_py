# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
data= pd.read_csv("../input/videogamesales/vgsales.csv")
data.head()
data.info()
data.describe()
print('genre unique values : ' , data['Genre'].unique())
print('Platform unique values : ' , data['Platform'].unique())
print('Publisher unique values :' , data['Publisher'].unique())

plt.figure()

data['Genre'].hist(figsize=(12,12))
plt.title("Genre")

data['Platform'].hist(figsize=(12,12))
plt.title("Platform")

data['Publisher'].hist(figsize=(12,12))
plt.title("Publisher")
data_corr = data.corr()
top_corr_features = data_corr.index
plt.figure(figsize=(12,12))
g= sb.heatmap(data_corr[top_corr_features],annot=True,cmap="RdYlGn")
plt.figure(figsize=(10,10))
plt.bar(data['Genre'],data['Global_Sales'])
plt.figure(figsize=(10,10))
plt.bar(data['Platform'],data['Global_Sales'])
#data_with_notNull = data.dropna(subset=['Publisher'])
data_ready = data.drop(['Name','Year','Rank','Publisher'],axis=1)
data_ready.info()
y = data_ready.Global_Sales  
X = data_ready.drop(['Global_Sales'],axis=1)

s = (X.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)

OH_cols_X = pd.DataFrame(OH_encoder.fit_transform(X[object_cols]))

print(OH_cols_X.info())
print(OH_cols_X.head())
X_train, X_valid, y_train, y_valid = train_test_split(OH_cols_X,y,train_size=0.7,test_size=0.3,random_state=0)
#cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
#print(cols_with_missing)
#print(X_train.info())
#my_imputer = SimpleImputer(strategy='most_frequent')
#imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
#imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(X_valid))

#imputed_X_train_columns = X_train.columns
#imputes_x_valid_columns = X_valid.columns
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

#LinearRegressionModel = LinearRegression(copy_X=True,n_jobs=-1)
#LinearRegressionModel.fit(X_train,y_train)
ridgeModel = Ridge()
from sklearn.model_selection import GridSearchCV
SelectedParameter = {}
alpha = [0.001, 0.01, 0.1, 1,2,3,4,5,6,7,8,9,9.1,9.5,9.8, 10,10,2,10.5,10.8,11,11.1,11.2,11.5,12,13,14,15,16,17,18,19,20, 100, 1000]
param_grid = dict(alpha=alpha)
GridSearchModel = GridSearchCV(ridgeModel,param_grid,cv=10,return_train_score=True)
grid_result = GridSearchModel.fit(X_train, y_train)
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
ridgeModel == Ridge(alpha = 20)
ridgeModel.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

print('train score :' , ridgeModel.score(X_train,y_train))
print('test score :' , ridgeModel.score(X_valid,y_valid))

y_predict_train_LR = pd.DataFrame(ridgeModel.predict(X_train))

LR_train_mae = mean_absolute_error(y_train,y_predict_train_LR)
LR_train_MSE = mean_squared_error(y_train,y_predict_train_LR)
LR_train_Median = median_absolute_error(y_train,y_predict_train_LR)

print("train MAE :",LR_train_mae)
print("train MSE :",LR_train_MSE)
print("train MedianAE :",LR_train_Median)


y_predict_valid_LR = pd.DataFrame(ridgeModel.predict(X_valid))

LR_valid_mae = mean_absolute_error(y_valid,y_predict_valid_LR)
LR_valid_MSE = mean_squared_error(y_valid,y_predict_valid_LR)
LR_valid_Median = median_absolute_error(y_valid,y_predict_valid_LR)

print("valid MAE :",LR_valid_mae)
print("valid MSE :",LR_valid_MSE)
print("valid MedianAE :",LR_valid_Median)
#print(y_train.head(5))
#print(y_predict_train.head(5))
plt.figure(figsize=(13,13))
plt.plot([n for n in range(0,50)],y_train[0:50])
plt.plot([n for n in range(0,50)],y_predict_train_LR[0:50])
#print(y_train.head(5))
#print(y_predict_train.head(5))
plt.figure(figsize=(13,13))
plt.plot([n for n in range(0,50)],y_valid[0:50])
plt.plot([n for n in range(0,50)],y_predict_valid_LR[0:50])
# KNN Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
knn_scores = []

for k in range (1,100):
    knn_regressor = KNeighborsRegressor(n_neighbors = k ,weights ='uniform',algorithm='auto')
    score = cross_val_score(knn_regressor,OH_cols_X,y,cv=10)
    knn_scores.append(score.mean())
    
plt.figure(figsize=(12,12))
plt.plot([k for k in range(1,100)],knn_scores,color='red')




from sklearn.neighbors import KNeighborsRegressor
KnnModel = KNeighborsRegressor(n_neighbors = 100,weights='uniform',algorithm='auto')
KnnModel.fit(X_train,y_train)
print('KNeighborsRegressorModel Train Score is : ' , KnnModel.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , KnnModel.score(X_valid, y_valid))
print('----------------------------------------------------')

y_predict_train_KNN = pd.DataFrame(KnnModel.predict(X_train))


KNN_train_mae = mean_absolute_error(y_train,y_predict_train_KNN)
KNN_train_MSE = mean_squared_error(y_train,y_predict_train_KNN)
KNN_train_Median = median_absolute_error(y_train,y_predict_train_KNN)

print("train MAE :",KNN_train_mae)
print("train MSE :",KNN_train_MSE)
print("train MedianAE :",KNN_train_Median)


y_predict_valid_KNN = pd.DataFrame(KnnModel.predict(X_valid))

KNN_valid_mae = mean_absolute_error(y_valid,y_predict_valid_KNN)
KNN_valid_MSE = mean_squared_error(y_valid,y_predict_valid_KNN)
KNN_valid_Median = median_absolute_error(y_valid,y_predict_valid_KNN)

print("valid MAE :",KNN_valid_mae)
print("valid MSE :",KNN_valid_MSE)
print("valid MedianAE :",KNN_valid_Median)
#print(y_train.head(5))
#print(y_predict_train.head(5))
plt.figure(figsize=(13,13))
plt.plot([n for n in range(0,50)],y_train[0:50])
plt.plot([n for n in range(0,50)],y_predict_train_KNN[0:50],color='red')
plt.plot([n for n in range(0,50)],y_predict_train_LR[0:50])
#print(y_train.head(5))
#print(y_predict_train.head(5))
plt.figure(figsize=(13,13))
plt.plot([n for n in range(0,50)],y_valid[0:50])
plt.plot([n for n in range(0,50)],y_predict_valid_KNN[0:50],color='red')
plt.plot([n for n in range(0,50)],y_predict_valid_LR[0:50])
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 500, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 20)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,'max_depth': max_depth}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
grid_result = rf_random.fit(X_train, y_train)
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)
from sklearn.ensemble import RandomForestRegressor 

RfModel = RandomForestRegressor(n_estimators =289,max_depth=15,random_state=0)
RfModel.fit(X_train,y_train)
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 500, num = 10)]
print(n_estimators)
print('RandomForestRegressorModel Train Score is : ' , RfModel.score(X_train, y_train))
print('RandomForestRegressorModel Test Score is : ' , RfModel.score(X_valid, y_valid))
print('----------------------------------------------------')

y_predict_train_RF = pd.DataFrame(RfModel.predict(X_train))


RF_train_mae = mean_absolute_error(y_train,y_predict_train_RF)
RF_train_MSE = mean_squared_error(y_train,y_predict_train_RF)
RF_train_Median = median_absolute_error(y_train,y_predict_train_RF)

print("train MAE :",RF_train_mae)
print("train MSE :",RF_train_MSE)
print("train MedianAE :",RF_train_Median)


y_predict_valid_RF = pd.DataFrame(RfModel.predict(X_valid))

RF_valid_mae = mean_absolute_error(y_valid,y_predict_valid_RF)
RF_valid_MSE = mean_squared_error(y_valid,y_predict_valid_RF)
RF_valid_Median = median_absolute_error(y_valid,y_predict_valid_RF)

print("valid MAE :",RF_valid_mae)
print("valid MSE :",RF_valid_MSE)
print("valid MedianAE :",RF_valid_Median)
plt.figure(figsize=(13,13))
plt.plot([n for n in range(0,50)],y_train[0:50])
plt.plot([n for n in range(0,50)],y_predict_train_KNN[0:50],color='red')
plt.plot([n for n in range(0,50)],y_predict_train_LR[0:50],color='blue')
plt.plot([n for n in range(0,50)],y_predict_train_RF[0:50],color='Green')

#print(y_train.head(5))
#print(y_predict_train.head(5))
plt.figure(figsize=(13,13))
plt.plot([n for n in range(0,50)],y_valid[0:50])
plt.plot([n for n in range(0,50)],y_predict_valid_KNN[0:50],color='red')
plt.plot([n for n in range(0,50)],y_predict_valid_LR[0:50],color='blue')
plt.plot([n for n in range(0,50)],y_predict_valid_RF[0:50],color='green')
MAE = [round(LR_valid_mae, 3),round(KNN_valid_mae,3),round(RF_valid_mae,3)]

plt.bar([0,1,2],MAE)

for xx,yy in zip([0,1,2],MAE):
    plt.text(xx,yy,yy)
MSE = [round(LR_valid_MSE, 3),round(KNN_valid_MSE,3),round(RF_valid_MSE,3)]
plt.bar([0,1,2],MSE)
for xx,yy in zip([0,1,2],MSE):
    plt.text(xx,yy,yy)
Median = [round(LR_valid_Median, 3),round(KNN_valid_Median,3),round(RF_valid_Median,3)]
plt.bar([0,1,2],Median)
for xx,yy in zip([0,1,2],Median):
    plt.text(xx,yy,yy)