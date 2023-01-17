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
trainingData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
competitionData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
trainingData.shape
competitionData.shape
trainingData.head()
trainingData.set_index('Id',inplace = True)
competitionData.set_index('Id',inplace = True)
trainingDataColumns = list(trainingData) 
  
for c in trainingDataColumns: 
    if trainingData[c].dtype == 'O':
        trainingData[c].fillna(value = 'Missing', inplace = True)
    else :
        trainingData[c].fillna(0, inplace = True)

trainingData.head()

competitionDataColumns = list(competitionData) 
  
for f in competitionDataColumns: 
    if competitionData[f].dtype == 'O':
        competitionData[f].fillna(value = 'Missing', inplace = True)
    else :
        competitionData[f].fillna(0, inplace = True)

competitionData.head()
trainingData = pd.get_dummies(trainingData)
competitionData = pd.get_dummies(competitionData)
trainingData.head()
trainingData.shape
competitionData.shape
sp = trainingData['SalePrice']
missingFeatures = list(set(trainingData.columns.values) - set(competitionData.columns.values))
trainingData = trainingData.drop(missingFeatures,axis=1)
missingFeatures = list(set(competitionData.columns.values) - set(trainingData.columns.values))
competitionData = competitionData.drop(missingFeatures,axis=1)
trainingData.shape
competitionData.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainingData, sp, random_state=0)
## Load ML algorithm

from sklearn.linear_model import Lasso
myModel = Lasso(alpha=298.4).fit(X_train,y_train)
submission = pd.DataFrame(myModel.predict(competitionData), columns=['SalePrice'], index = competitionData.index)
display(submission.head())
submission.to_csv("submission-lv.csv")
