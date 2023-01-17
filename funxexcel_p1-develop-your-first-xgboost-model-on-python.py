import pandas as pd

import numpy as numpy

import xgboost as xgb #contains both XGBClassifier and XGBRegressor
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.info()
#Get Target data 

y = data['Outcome']



#Load X Variables into a Pandas Dataframe with columns 

X = data.drop(['Outcome'], axis = 1)
X.head()
#Check size of data

X.shape
X.isnull().sum()

#We do not have any missing values
xgbModel = xgb.XGBClassifier() #max_depth=3, n_estimators=300, learning_rate=0.05
xgbModel.fit(X,y)
print (f'Accuracy - : {xgbModel.score(X,y):.3f}')