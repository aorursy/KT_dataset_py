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
#Importing the neccessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.preprocessing import Binarizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
#loading and visualizing the Dataset

dataset = pd.read_csv('../input/machine-learning-for-diabetes-with-python/diabetes_data.csv')

dataset.head()
#description of dataset

dataset.describe()
#selection and splitting of data

data = dataset.iloc[:,0:8]

outcome = dataset.iloc[:,8]

x,y = data,outcome
#distribution of dataset into training and testing sets

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
#counting positive and negative values

print(y_test.value_counts())

#mean of testng distn

print(1- y_test.mean())
#Parameter evaluation with gsc validation

#Using Grid Search Cross Validation for evaluating the best parameters

gbe = GradientBoostingClassifier(random_state=0)

parameters={

    'learning_rate': [0.05, 0.1, 0.5],

    'max_features' : [0.5, 1],

    'max_depth' : [3, 4, 5]

}

gridsearch=GridSearchCV(gbe,parameters,cv=100,scoring='roc_auc')

gridsearch.fit(x,y)

print(gridsearch.best_params_)

print(gridsearch.best_score_)
#adjusting development threshlod

gbi = GradientBoostingClassifier(learning_rate=0.05,max_depth=3,max_features=0.5,random_state=0)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

gbi.fit(x_train,y_train)
#storing the prediction

yprediction = gbi.predict_proba(x_test)[:,1]
#plotting the predictions

plt.hist(yprediction,bins=10)

plt.xlim(0,1)

plt.xlabel("Predicted Probabilities")

plt.ylabel("Frequency")
#Score of Gradient Boosting Classifier

round(roc_auc_score(y_test,yprediction),5)
#using random forest classification (scr of rfc)

from sklearn.ensemble import RandomForestClassifier

rmfr = RandomForestClassifier()

rmfr.fit(x_train, y_train)

y_pred = rmfr.predict(x_test)

accuracyrf = round(accuracy_score(y_pred, y_test), 5)

accuracyrf
#Score of XGBoost Classifier 

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]

accuracy = round(accuracy_score(y_test, predictions),5)

accuracy