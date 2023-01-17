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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

df.head()
df.describe()
df.info()
df.isnull().sum()
df.isnull().any().any()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error



corr = df.corr()

#print(corr['rating'])

sns.heatmap(corr)
df['class'].value_counts()
numerical_features = ['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

#categorical_features = ['type']

#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()

#categorical_features=le.fit_transform(categorical_features)

#print(categorical_features.dtype)

#print(df['feature1'].dtype)

X_train = df[numerical_features]

#X_train['type']=le.fit_transform(X_train['type'])

y_train = df["class"]
print(X_train.shape)

print(y_train.shape)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBClassifier

#from xgboost.sklearn import XGBClassifier

#TODO

clf = XGBClassifier()       #Initialize the classifier object

#train_x = X_train.values

#parameters = {'n_estimators':[10,50,100,150,200]}    #Dictionary of parameters

parameters = {

        'colsample_bytree': [0.8,0.9, 1.0],

        'max_depth': [3,4],

        'n_estimators':[50,60,70,80,90,100]

        }

scorer = make_scorer(accuracy_score, greater_is_better = True)#Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(best_clf)

#unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_train)        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_opd = accuracy_score(y_train, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_opd))
xg_best=best_clf
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesClassifier

#from xgboost.sklearn import XGBClassifier

#TODO

clf = ExtraTreesClassifier()       #Initialize the classifier object

#train_x = X_train.values

#parameters = {'n_estimators':[10,50,100,150,200]}    #Dictionary of parameters

parameters = {'n_estimators':[1000,1500,2000,2500,3000,3500,4000],'min_samples_leaf':[1,2],'max_features':['sqrt'],'min_samples_split':[2],'max_depth':[25,30],'bootstrap':[False]}

#scorer = make_scorer(accuracy_score, greater_is_better = True)#Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(best_clf)

#unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_train)        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_opd = accuracy_score(y_train, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_opd))

et_best=best_clf
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

#from xgboost.sklearn import XGBClassifier

#TODO

clf = RandomForestClassifier()       #Initialize the classifier object

#train_x = X_train.values

#parameters = {'n_estimators':[10,50,100,150,200]}    #Dictionary of parameters

parameters = {'n_estimators':[800,950,1000,1100,1200],'min_samples_leaf':[1,2],'max_features':['sqrt'],'min_samples_split':[2],'bootstrap':[False,True]}

scorer = make_scorer(accuracy_score, greater_is_better = True)#Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(best_clf)

#unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_train)        #Same, but use the best estimator



#acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_opd = accuracy_score(y_train, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_opd))

rf_best=best_clf
et=ExtraTreesClassifier(n_estimators=1500,min_samples_leaf=1,min_samples_split=2,max_features='sqrt',max_depth=30,bootstrap=False)

et.fit(X_train,y_train)
et2=ExtraTreesClassifier(n_estimators=1500,min_samples_leaf=2,min_samples_split=2,max_features='sqrt',max_depth=30,bootstrap=False)

et2.fit(X_train,y_train)
et3=ExtraTreesClassifier(n_estimators=1000,min_samples_leaf=1,min_samples_split=2,max_features='sqrt',max_depth=30,bootstrap=False)

et3.fit(X_train,y_train)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

estimators=[('xgb', xg_best),('rf',rf_best),('et',et_best)]

eclf1 =VotingClassifier(estimators,voting='soft')

eclf1=eclf1.fit(X_train,y_train)
df4=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

#df4.fillna(value=df.mean(),inplace=True)

df4.isnull().sum()
df4.isnull().sum()
df4.isnull().any().any()
numerical_features = ['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

#categorical_features = ['type']

#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()

#categorical_features=le.fit_transform(categorical_features)

#print(categorical_features.dtype)

#print(df['feature1'].dtype)

X_test = df4[numerical_features]

#X_train['type']=le.fit_transform(X_train['type'])

#y_train = df["class"]
print(X_train)

print(X_test)
opt=et.predict(X_test)

#opt=clf.predict(X_test)

print(len(opt))

print(opt)
opt2=et2.predict(X_test)

#opt=clf.predict(X_test)

print(len(opt2))

print(opt2)
opt3=eclf1.predict(X_test)

#opt=clf.predict(X_test)

print(len(opt2))

print(opt2)
opt4=et3.predict(X_test)

#opt=clf.predict(X_test)

print(len(opt4))

print(opt4)
df3=pd.DataFrame({"id":df4["id"],"class":opt})

df3.to_csv('z24.csv',index=False)
df3=pd.DataFrame({"id":df4["id"],"class":opt2})

df3.to_csv('t1.csv',index=False)
df3=pd.DataFrame({"id":df4["id"],"class":opt3})

df3.to_csv('t2.csv',index=False)
df3=pd.DataFrame({"id":df4["id"],"class":opt4})

df3.to_csv('t4.csv',index=False)