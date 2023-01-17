# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

data.info()
import seaborn as sns

from pandas.plotting import scatter_matrix

import matplotlib.cm as cm

import matplotlib.pyplot as plt

data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

y = pd.DataFrame(data = data['diagnosis'])   

list = ['Unnamed: 32','id','diagnosis']

x = data.drop(list,axis = 1 )

x.head()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
rem = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

data_new = x.drop(rem, axis=1)

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(data_new.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
data_new['diagnosis'] = y

corrmat = data_new.corr().abs()

top_corr_features = corrmat.index[abs(corrmat["diagnosis"])>0.1]

plt.figure(figsize=(10,10))

g = sns.heatmap(data_new[top_corr_features].corr(),annot=True,cmap="RdYlGn")
req = ['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean', 'area_se', 'concavity_se',

      'smoothness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst']

data_new2 = data_new[req]

data_new = data_new.drop(['diagnosis'], axis = 1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(data_new2, y, test_size=0.3, random_state=42)



#random forest classifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)      

clr_rf = clf_rf.fit(x_train,y_train)



ac = accuracy_score(y_test,clf_rf.predict(x_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf.predict(x_test))

sns.heatmap(cm,annot=True,fmt="d")
data_final = pd.read_csv('../input/data.csv')

y_final = data_final['diagnosis'].map({'M':0, 'B':1})

data_final = data_final[req]

data_final.info()
from sklearn.model_selection import train_test_split

x_final = data_final

x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2, random_state=42)

x_train.shape
from sklearn.model_selection import RandomizedSearchCV



criterion =['mse', 'friedman_mse', 'mae']

splitter =['best', 'random']

max_depth = [None, 1, 11, 21, 31]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

max_features = ['auto', 'sqrt', 'log2']



random_grid = {'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'criterion' :criterion,

              'splitter' : splitter}



print(random_grid)



model_test = DecisionTreeRegressor(random_state=43)    

model_random = RandomizedSearchCV(estimator = model_test, param_distributions = random_grid, n_iter = 100, cv = 3, random_state = 42, n_jobs = -1)

model_random.fit(x_train, y_train)

model_random.best_params_
clf_rf = RandomForestClassifier(random_state=43, n_estimators = 20, min_samples_split=2, min_samples_leaf=2,

                               max_features='sqrt', max_depth=21,bootstrap=False)      

clr_rf = clf_rf.fit(x_train,y_train)



print(r2_score(y_test,clf_rf.predict(x_test)))

print(mean_squared_error(y_test, clf_rf.predict(x_test)))

from sklearn.ensemble import ExtraTreesClassifier



etc_test = ExtraTreesClassifier(random_state=43, n_estimators=300, min_samples_split=2,min_samples_leaf=1,

                               max_features='sqrt',max_depth=31, bootstrap=True)

etc_test = etc_test.fit(x_train,y_train)



print(r2_score(y_test,etc_test.predict(x_test)))

print(mean_squared_error(y_test, etc_test.predict(x_test)))
from sklearn.ensemble import ExtraTreesRegressor



etr_test = ExtraTreesRegressor(random_state=43,n_estimators=100, min_samples_split=2, min_samples_leaf=1,

                               max_features='auto',max_depth = None,bootstrap=False)

etr_test = etr_test.fit(x_train,y_train)



print(r2_score(y_test,etr_test.predict(x_test)))

print(mean_squared_error(y_test, etr_test.predict(x_test)))
from sklearn.ensemble import RandomForestRegressor



rfr_test = RandomForestRegressor(random_state=43,n_estimators=200,min_samples_split=2,min_samples_leaf=1,

                                max_features='log2',max_depth=11, bootstrap=False)

rfr_test = rfr_test.fit(x_train,y_train)



print(r2_score(y_test,rfr_test.predict(x_test)))

print(mean_squared_error(y_test, rfr_test.predict(x_test)))
from sklearn.tree import DecisionTreeClassifier



dtc_test = DecisionTreeClassifier(random_state=43,min_samples_leaf=8)

dtc_test = dtc_test.fit(x_train,y_train)



print(r2_score(y_test,dtc_test.predict(x_test)))

print(mean_squared_error(y_test, dtc_test.predict(x_test)))
from sklearn.tree import DecisionTreeRegressor



dtr_test = DecisionTreeRegressor(random_state=43,splitter='best',min_samples_split=2,min_samples_leaf=8,

                                max_features='auto', max_depth=11, criterion='mse')

dtr_test = dtr_test.fit(x_train,y_train)



print(r2_score(y_test,dtr_test.predict(x_test)))

print(mean_squared_error(y_test, dtr_test.predict(x_test)))
from sklearn import ensemble

gbr = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 7, min_samples_split = 8,

          learning_rate = 0.1, loss = 'ls')

gbr = gbr.fit(x_train,y_train)



print(r2_score(y_test,gbr.predict(x_test)))

print(mean_squared_error(y_test, gbr.predict(x_test)))
from mlxtend.regressor import StackingCVRegressor

rfc = RandomForestClassifier(random_state=43, n_estimators = 20, min_samples_split=2, min_samples_leaf=2,

                               max_features='sqrt', max_depth=21,bootstrap=False)   

etc = ExtraTreesClassifier(random_state=43, n_estimators=300, min_samples_split=2,min_samples_leaf=1,

                               max_features='sqrt',max_depth=31, bootstrap=True)

etr = ExtraTreesRegressor(random_state=43,n_estimators=100, min_samples_split=2, min_samples_leaf=1,

                               max_features='auto',max_depth = None,bootstrap=False)

rfr = RandomForestRegressor(random_state=43,n_estimators=200,min_samples_split=2,min_samples_leaf=1,

                                max_features='log2',max_depth=11, bootstrap=False)

dtc = DecisionTreeClassifier(random_state=43,min_samples_leaf=8)

dtr = DecisionTreeRegressor(random_state=43,splitter='best',min_samples_split=2,min_samples_leaf=8,

                                max_features='auto', max_depth=11, criterion='mse')

gbr = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 7, min_samples_split = 8,

          learning_rate = 0.1, loss = 'ls')

stack_gen = StackingCVRegressor(regressors=(rfc, etr, rfr, dtc, dtr),

                                meta_regressor=dtr,

                                use_features_in_secondary=True)



stack_gen_model = stack_gen.fit(x_train,y_train)



print(r2_score(y_test,stack_gen_model.predict(x_test)))

print(mean_squared_error(y_test, stack_gen_model.predict(x_test)))
def blend_models_predict(X):

    return ((0.05 * etc.predict(X)) + \

            (0.05 * gbr.predict(X)) + \

            (0.1 * rfc.predict(X)) + \

            (0.1 * rfr.predict(X)) + \

            (0.1 * dtc.predict(X)) + \

            (0.2 * dtr.predict(X)) + \

            (0.1 * etr.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))



etc_model = etc.fit(x_train,y_train)

gbr_model = gbr.fit(x_train,y_train)

rfc_model = rfc.fit(x_train,y_train)

rfr_model = rfr.fit(x_train,y_train)

dtc_model = dtc.fit(x_train,y_train)

dtr_model = dtr.fit(x_train,y_train)

etr_model = etr.fit(x_train,y_train)



print(r2_score(y_test,blend_models_predict(x_test)))

print(mean_squared_error(y_test, blend_models_predict(x_test)))