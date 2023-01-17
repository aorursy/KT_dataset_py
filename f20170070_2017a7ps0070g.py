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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)
train = pd.read_csv('../input/eval-lab-1-f464-v2/train.csv')

test = pd.read_csv('../input/eval-lab-1-f464-v2/test.csv')
train.fillna(value=train.mean(),inplace = True)

train.drop(['feature4','feature7','feature8','feature11','type'],axis = 1)
X = train[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']]

y = train['rating']
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42) 
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_train[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']] = scaler.fit_transform(X_train[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']])

X_val[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']] = scaler.transform(X_val[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']])  
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error

from math import sqrt



#TODO

clf = RandomForestClassifier()        #Initialize the classifier object



parameters = {'n_estimators':[10,50,100,105,110,115,120,125,130,135,140,145,150]}    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = RandomizedSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator



unoptimized_predictions = [round(a) for a in unoptimized_predictions]

optimized_predictions = [round(a) for a in optimized_predictions]



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))

rms = sqrt(mean_squared_error(y_val, optimized_predictions))

rms
from sklearn.linear_model import LinearRegression



reg_lr = LinearRegression().fit(X_train,y_train)

y_pred_lr = reg_lr.predict(X_val)



y_pred_lr = [round(a) for a in y_pred_lr]



rms = sqrt(mean_squared_error(y_val, y_pred_lr))

rms

X_test = test[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']]

X_test.fillna(X_test.mean(),inplace = True)



X_test[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']] = scaler.transform(X_test[['feature1','feature2','feature3','feature5','feature6','feature9','feature10']])
optimized_predictions = best_clf.predict(X_test)

optimized_predictions = [round(a) for a in optimized_predictions]

optimized_predictions
submission = pd.DataFrame({'id':test['id'],'rating':optimized_predictions})

submission.to_csv('sub6.csv',index=False)