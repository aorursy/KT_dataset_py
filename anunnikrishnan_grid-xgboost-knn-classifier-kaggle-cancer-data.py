# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

## pip install xgboost



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import xgboost as xgb









# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data  = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
data.columns

data.drop(['id','Unnamed: 32'],axis =1, inplace = True)
data.columns
data['diagnosis'].unique()

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
data
# standard code to check for null values in columns in data

print(data.isnull().any())

# show how many null values there.

print(data.isnull().sum())

#from sklearn.neighbors import KNeighborsClassifier

#from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



#from sklearn.ensemble import RandomForestClassifier

#my_first_model = RandomForestClassifier()

#neigh.fit(samples)







from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



### Automated Hyper parameter Tuning





# A parameter grid for XGBoost



paramGrid = { 'learning_rate': [0.1,0.001] , 'max_depth': [4,10],'n_estimators': [500,5000]}



"""

fit_params={"early_stopping_rounds":100, 

            "eval_metric" : "mae", 

            "eval_set" : [[testX, testY]]}

"""

"""

combination 1: 

max_depth = 4 and n_estimator = 100

combination 2:

max_depth = 4 and n_estimator = 200

combination 3:

max_depth = 5 and n_estimator = 100

combination 4:

max_depth = 5 and n_estimator = 200

combination 5:

max_depth = 6 and n_estimator = 100

max_depth = 6 and n_estimator = 200



"""



# cv = None, default 3 fold cross validation

#my_first_model = LGBMClassifier(max_depth=4,n_estimator=100)

my_algo =['xgboost','lightgbm']



my_first_model = LGBMClassifier()



### calling 

#mysearch = GridSearchCV(my_first_model, paramGrid, verbose=1 ,cv=5)

mysearch = RandomizedSearchCV(my_first_model, paramGrid, verbose=1 ,cv=5)



print(mysearch)
#print(my_first_model)
X = data.drop('diagnosis',axis =1)

y = data['diagnosis']
X
y
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 3)
y_test
# pass features and dependent varaible

# fit means model training

#my_first_model.fit(X_train, y_train)



mysearch.fit(X_train,y_train)





# predict

prediction = mysearch.best_estimator_.predict(X_test)
mysearch.best_estimator_
print(mysearch.best_params_)
#prediction = my_first_model.predict(X_test)

print(prediction)

print(len(prediction))
print(prediction)
X_test
from sklearn.metrics import accuracy_score, confusion_matrix , f1_score , precision_score, recall_score

# pass actual values , predicted values

# Validation accuracy

accuracy_score(y_test,prediction)
confusion_matrix(y_test,prediction)
print(f1_score(y_test,prediction))

print(precision_score(y_test,prediction))

print(recall_score(y_test,prediction))

print(accuracy_score(y_test,prediction))
#importance = my_first_model.feature_importances_
#features = X.columns
#feature_score = pd.DataFrame(list(zip(features,importance)),columns = ['features','importance'])
#feature_score.sort_values('importance',ascending = False)#
#train_prediction = my_first_model.predict(X_train)
#train_prediction

# Acutal is y_test

# training accuracy

#accuracy_score(y_train,train_prediction)
### Changing K value of algorithm is called Hyper parameter tuning



## Algorith Used : KNN Classifier



#Models:

##### iteration 1: 0.9322 k = 2, model 1 

##### iteration 2: 0.9298 K= 7, model 2

#### iteration 3: 0.9181 K = 4,model 3



### Random Forest : 0.953



### Xgboost: 0.959

### LightGBM:  0.959

"""



Steps to follow:

 1. Fetch data 

 2. Explore data/ Panda profiling

 3. Decide the model type( regression or classification)

 4.Feature importance ( train model and find best feature) apply model with all featues

 5.Validate model - if not good remove / add more features and then re-validate.

 6.Change and validate with different algorithm

 7.Auto hyper paramter tuning ( grid search)- this gives += 2% increase maybe

 

 

 

















"""