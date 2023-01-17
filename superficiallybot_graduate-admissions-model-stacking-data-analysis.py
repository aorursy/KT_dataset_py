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
import pandas as pd

import numpy as np



# for data visualizations



import matplotlib.pyplot as plt

import seaborn as sns





import os

import pickle

import joblib
data = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv', index_col = 'Serial No.')

data2 = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', index_col = 'Serial No.')



print('Shape of data1 : ', data.shape)

print('Shape of data2 : ', data2.shape)
data = pd.concat([data, data2], axis = 0)
data.head()
data.shape
data.columns
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Chance of Admit '], axis = 1), data['Chance of Admit '], test_size = 0.2, random_state = 21)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train.shape
X_test.shape
X_train
X_test
joblib.dump(sc, 'scaler.save')
scaler_loaded = joblib.load('scaler.save')
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size= 0.2, random_state = 21)
print(X_train[:5])

print('-' * 10)

print(X_train.shape)
print(X_val[:5])

print('-' * 10)

print(X_val.shape)
print(y_train[:5])

print('-' * 10)

print(y_train.shape)
print(y_val[:5])

print('-' * 10)

print(y_val.shape)
print(X_test[:5])

print('-' * 10)

print(X_test.shape)
print(y_test[:5])

print('-' * 10)

print(y_test.shape)
# work on data distribution for later stages of model training
from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



linreg = LinearRegression()

linreg.fit(X_train, y_train)



linreg_pred = linreg.predict(X_test)



mse = mean_squared_error(y_test, linreg_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, linreg_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from sklearn.svm import SVR



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



svr = SVR(kernel = 'linear')

svr.fit(X_train, y_train)



svr_pred = svr.predict(X_test)



mse = mean_squared_error(y_test, svr_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, svr_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



rfr = RandomForestRegressor(100)

rfr.fit(X_train, y_train)



rfr_pred = rfr.predict(X_test)



mse = mean_squared_error(y_test, rfr_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, rfr_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from xgboost.sklearn import XGBRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



xgb = XGBRegressor()

xgb.fit(X_train, y_train)



xgb_pred = xgb.predict(X_test)



mse = mean_squared_error(y_test, xgb_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, xgb_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from sklearn.ensemble import ExtraTreesRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



etr = ExtraTreesRegressor()

etr.fit(X_train, y_train)



etr_pred = etr.predict(X_test)



mse = mean_squared_error(y_test, etr_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, etr_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
"""polyreg = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())

polyreg.fit(X_train,y_train)

polyreg.score(X_test, y_test)"""
from sklearn import linear_model

from sklearn import svm

from sklearn import neighbors

from sklearn import tree

# ML Models



classifiers_single_model = [

    linear_model.Ridge(alpha = 0.5),

    linear_model.Lasso(alpha = 0.1),

    #linear_model.LassoLars(alpha = 0.1),

    linear_model.BayesianRidge(),

    linear_model.SGDRegressor(max_iter = 1000, tol = 1e-3),

    svm.SVR(),

    neighbors.KNeighborsRegressor(),

    tree.DecisionTreeRegressor(),

    linear_model.LinearRegression(),

    linear_model.Lasso(alpha = 0.12),

    linear_model.Ridge(alpha = 0.55),

    linear_model.Ridge(alpha = 0.6),

    linear_model.Ridge(alpha = 0.65),

    linear_model.Ridge(alpha = 0.7),

    make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())  

]
len(classifiers_single_model)
X_test.shape
features_level_1_train = np.zeros((144,14))

features_level_1_test = np.zeros((180,14))

for i, clf in enumerate(classifiers_single_model):

    print(clf.fit(X_train, y_train))

    print(clf.score(X_val, y_val))

    preds = clf.predict(X_val)

    #print(preds.shape)

    features_level_1_train[:, i] = preds

    # preds for test data

    preds_test = clf.predict(X_test)

    #features_level_1_test = clf.predict(X_test)

    features_level_1_test[:, i] = preds_test

    pickle.dump(clf, open(f'model{i+1}' + '.pkl', 'wb'))

    
features_level_1_train
len(features_level_1_train)
y_val
len(y_val)
stage_1_ensemble_5_models = [

    make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression()),

    neighbors.KNeighborsRegressor(),

        tree.DecisionTreeRegressor(),

            linear_model.SGDRegressor(max_iter = 1000, tol = 1e-3),

    svm.SVR()

]
val_size = len(features_level_1_train) * 0.4
val_size = np.ceil(val_size)

val_size
features_level_1_train, features_level_1_val, y_level_1_train, y_level_1_val = train_test_split(features_level_1_train, y_val, test_size = 0.4, random_state = 21)
features_level_1_train.shape
features_level_1_val.shape
y_level_1_train.shape
y_level_1_val.shape
features_level_2_train = np.zeros((58,5))

features_level_2_test = np.zeros((180, 5))

for i, clf in enumerate(stage_1_ensemble_5_models):

    print(clf.fit(features_level_1_train, y_level_1_train))

    #print(clf.score(X_test, y_test))

    print(clf.score(features_level_1_val, y_level_1_val))

    preds = clf.predict(features_level_1_val)

    print(preds.shape)

    features_level_2_train[:, i] = preds

    

    preds_test = clf.predict(features_level_1_test)

    features_level_2_test[:, i] = preds_test

    

    pickle.dump(clf, open(f'ensemble_model{i+1}' + '.pkl', 'wb'))

    

    
features_level_2_train
features_level_2_train.shape
y_level_1_val.shape
features_level_2_test
features_level_2_test.shape
# CatBoostRegressor, ExtraTreesRegressor, XGBRegressor, RandomForestRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb



stage_2_models = [

    CatBoostRegressor(iterations = 500, depth = 4),

    ExtraTreesRegressor(),

    RandomForestRegressor()    

]



features_level_22_train, features_level_22_val, y_level_2_train, y_level_2_val = train_test_split(features_level_2_train, y_level_1_val, test_size = 0.4, random_state = 21)
features_level_22_train
features_level_22_train.shape
y_level_2_train
y_level_2_train.shape
features_level_22_val
features_level_22_val.shape
y_level_2_val
y_level_2_val.shape
features_level_3_train = np.zeros(shape = (24, 3))

features_level_3_test = np.zeros(shape = (180,3))



for i, clf in enumerate(stage_2_models):

    print(clf.fit(features_level_22_train, y_level_2_train))

    #print(clf.score(X_test, y_test))

    print(clf.score(features_level_22_val, y_level_2_val))

    preds = clf.predict(features_level_22_val)

    print(preds.shape)

    features_level_3_train[:, i] = preds

    

    preds_test = clf.predict(features_level_2_test)

    print(preds_test.shape)

    features_level_3_test[:, i] = preds_test

    pickle.dump(clf, open(f'stage_2_model{i+1}'+'.pkl','wb'))
features_level_3_train
features_level_3_train.shape
y_level_2_val.shape
features_level_3_test.shape
y_test.shape
meta_model = linear_model.LinearRegression(n_jobs= -1)

meta_model.fit(features_level_3_train, y_level_2_val)
meta_model.score(features_level_3_test, y_test)
import pickle
pickle.dump(meta_model, open('meta_model.pkl', 'wb'))
dir(meta_model)
# saving the ml model in pickle format
files = [file for file in os.listdir('/kaggle/working') if file.endswith('.pkl')]
files
len(files)