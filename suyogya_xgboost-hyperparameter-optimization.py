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
#Importing Libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
#Importing dataset

dataset = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

df= dataset.copy()
dataset.head()
dataset.info()
dataset.describe()
dataset.isnull().sum() #There is no null values in our dataset
dataset.isna().sum() #There is no NAN ( Not available ) values in our dataset
import seaborn as sns

sns.countplot(data=dataset,x=dataset['Exited']) #We can see out of 10k records, 8k -> 0 NOT EXITED and 2k -> 1 YES EXITED
#Lets find out the correlation of features using heatmap

corrmat = dataset.corr()

top_corr_feature = corrmat.index

plt.figure(figsize=(20,20))

g = sns.heatmap(dataset[top_corr_feature].corr(),annot=True,cmap='RdYlGn')
#Lets select the features

X = dataset.iloc[:,3:-1]

y = dataset.iloc[:,-1]
X #Here we have two categorical features : Geography and Gender which needs to be converted to numeric values
y
#Using One Hot Encoding technique for categorical feature

X["Geography"].unique()
X_Geo = pd.get_dummies(data=dataset['Geography'])

X_Geo
X_Gender = pd.get_dummies(data=dataset['Gender'],drop_first=True)

X_Gender
X = pd.concat([X,X_Geo,X_Gender],axis=1)
X #Concatenation of these with our dataset
#Dropping the columns GEOGRAPHY and GENDER

X = X.drop(["Geography","Gender"],axis=1)
X #Every columns have numeric values now
X.info()
#Splitting the dataset into test and train set 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=(np.random))
#Splitting the dataset into test and train set 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=(np.random))
#Create a model of XGBoost

from xgboost import XGBClassifier

XGBclassifier = XGBClassifier()

XGBclassifier = XGBclassifier.fit(X_train,y_train)
XGBclassifier #we have used default hyperparameters for now.
y_pred = XGBclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print("Confusion Matrix : \n")

print(confusion_matrix(y_test,y_pred))
print("Classification Report : \n")

print(classification_report(y_test,y_pred))
XGBoost_Model_Without_Optimization = round(accuracy_score(y_test,y_pred)*100,2)

XGBoost_Model_Without_Optimization
params = {

    "booster" : ["gbtree","gblinear","dart"],

    "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],

    "max_depth" : [3,4,5,6,7,8,9,10,11,12,15],

    "min_child_weight" : [1,3,5,7],

    "gamma" : [0.0,0.1,0.2,0.3,0.4],

    "colsample_bytree" : [0.3,0.4,0.5,0.7]

}
#Using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

from datetime import datetime
#Lets create function to capture time

def timer(start_time=None):

    if not start_time:

        start_time= datetime.now()

        return start_time

    elif start_time:

        thour,temp_sec = divmod((datetime.now() - start_time).total_seconds(),3600)

        tmin, tsec = divmod(temp_sec,60)

        print("\n Time Taken : %i hours %i minutes and %s seconds. "%(thour,tmin,round(tsec,2)))
XGBclassifier = XGBClassifier()
random_search = RandomizedSearchCV(XGBclassifier,param_distributions=params,n_iter=5,scoring="roc_auc",n_jobs=-1,cv=5,verbose=3)
start_time = timer(None)

random_search.fit(X_test,y_test)

timer(start_time)
random_search.best_estimator_
random_search.best_params_
#this is best hyperparameters optimization for XGBClassifier

XGBClassifier = XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0.3,

              learning_rate=0.1, max_delta_step=0, max_depth=6,

              min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
XGBClassifer = XGBClassifier.fit(X_train,y_train)
y_pred = XGBClassifier.predict(X_test)

print("Confusion Matrix : \n")

print(confusion_matrix(y_test,y_pred))

print("===============")

print("Classification Report : \n")

print(classification_report(y_test,y_pred))
XGBoost_Model_With_RandomizedSearchCV_Optimization = round(accuracy_score(y_test,y_pred)*100,2)

XGBoost_Model_With_RandomizedSearchCV_Optimization
params={

    'max_depth': [2], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env

    'subsample': [0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],

    'colsample_bytree': [0.5], #[0.5,0.6,0.7,0.8],

    'n_estimators': [1000], #[1000,2000,3000]

    'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]

}
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
GridSearchCV = GridSearchCV(XGBClassifier,params,cv=5,scoring="roc_auc",n_jobs=1,verbose=2)
start_time = timer(None)

GridSearchCV.fit(X_test,y_test)

timer(start_time)
GridSearchCV.best_params_
GridSearchCV.best_estimator_
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
XGBClassifier= XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=2,

              min_child_weight=1, missing=None, n_estimators=1000, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0.03, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=0.6, verbosity=1)
XGBClassifier= XGBClassifier.fit(X_train,y_train)
y_pred = XGBClassifier.predict(X_test)

print("Confusion Matrix : \n")

print(confusion_matrix(y_test,y_pred))

print("===============")

print("Classification Report : \n")

print(classification_report(y_test,y_pred))
XGBoost_Model_With_GridSearchCV_Optimization =round(accuracy_score(y_test,y_pred)*100,2)

XGBoost_Model_With_GridSearchCV_Optimization
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

XGBClassifier = XGBClassifier()

score = cross_val_score(XGBClassifier,X,y,cv=10)
score
K_fold_CV_Score = round(score.mean()*100)

K_fold_CV_Score
from sklearn.model_selection import StratifiedKFold as skf

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

XGBClassifier = XGBClassifier()

X.shape,y.shape
X.iloc[9999]
#Lets observe Stratified K Fold

from sklearn.metrics import accuracy_score

accuracy =[]



skf = skf(n_splits=5,random_state=None)

skf.get_n_splits(X,y)



for train_index,test_index in skf.split(X,y):

    print("Train : ",train_index," Validation : ",test_index)

    X1_train,X1_test = X.iloc[train_index], X.iloc[test_index]

    y1_train,y1_test = y.iloc[train_index], y.iloc[test_index]

    

    XGBClassifier.fit(X1_train,y1_train)

    prediction = XGBClassifier.predict(X1_test)

    score = accuracy_score(prediction,y1_test)

    accuracy.append(score)



print(accuracy)  

    
Stratified_K_Fold_Score = round(np.array(accuracy).mean()*100,2)

Stratified_K_Fold_Score
import pandas as pd

model = pd.DataFrame({

    'Model' : ['XGBoost_Model_Without_Optimization',

               'XGBoost_Model_With_RandomizedSearchCV_Optimization',

               'XGBoost_Model_With_K_fold_CrossValidation',

               'XGBoost_Model_With_Stratified_K_Fold_CrossValidation'],

    'Score' : [76,83,86,91]  

})

# 'Score' : [XGBoost_Model_Without_Optimization,

#                 XGBoost_Model_With_RandomizedSearchCV_Optimization,

#                 K_fold_CV_Score,

#                 Stratified_K_Fold_Score]
model
model_sorted= model.sort_values(by='Score',ascending=False)

model_sorted
#Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

x = model_sorted['Model']

y = model_sorted['Score']

plt.barh(x,y)



plt.xlim(50,100)

plt.tick_params(labelsize=12)

plt.title('Hyperparameters Optimization of XGBOOST using K fold and Stratified K fold cross validation')

plt.xlabel('Accuracy Score')

plt.ylabel('Models')

plt.show()