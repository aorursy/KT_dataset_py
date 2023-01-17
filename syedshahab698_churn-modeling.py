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
data = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
data.head()
import matplotlib.pyplot as plt

import seaborn as sns

def summary(data):

    print('Shape: ' , data.shape)

    return( pd.DataFrame({ "Dtypes ":data.dtypes , 

                           "NAs":data.isnull().sum() ,

                           "uniques":data.nunique() ,

                            "Levels":[ data[i].unique() for i in data.columns]}))
summary(data)
data.columns =  data.columns.str.lower()
data.columns
#dropping unnecessary columns

data.drop(['rownumber','customerid','surname'] ,axis=1 , inplace=True)
summary(data)
# not splitting now because i am not doing any analysis just dropping columns r dummifying

data = pd.get_dummies(data , columns=data.select_dtypes('object').columns)
summary(data)
for i in data.select_dtypes('uint8').columns:

    data[i] = data[i].astype('int64')
summary(data)
# splitting and model building

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC

x= data.drop('exited' , axis=1 )

y=data['exited']

y = y.astype('category')

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
from sklearn.metrics import accuracy_score , recall_score

def model_building(model):

   

    model.fit(xtrain,ytrain)

    train_pred = model.predict(xtrain)

    test_pred = model.predict(xtest)

    print("============================================================================= \n ",model , "\n===========================================================================" )

    print('Train Accuracy and Recall :' ,accuracy_score(ytrain ,train_pred) , recall_score(ytrain , train_pred , pos_label=1) )

    print('Test Accuracy and Recall :' ,accuracy_score(ytest ,test_pred) , recall_score(ytest , test_pred ,pos_label=1) )
import warnings as w

w.filterwarnings('ignore')

models = [ LogisticRegressionCV(),DecisionTreeClassifier() ,RandomForestClassifier() , GradientBoostingClassifier() ,AdaBoostClassifier() ,

           XGBClassifier() ,SVC()]

for i in models:

    model_building(i)