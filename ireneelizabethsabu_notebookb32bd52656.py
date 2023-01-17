# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
train.head()
data = train.append(test,sort=False)

data = data.dropna(axis=1, how='any', thresh = 1000)
data = data.fillna(data.mean()) 
data = pd.get_dummies(data)
covarianceMatrix = data.corr()
listOfFeatures = [i for i in covarianceMatrix]
setOfDroppedFeatures = set() 
for i in range(len(listOfFeatures)) :
    for j in range(i+1,len(listOfFeatures)): #Avoid repetitions 
        feature1=listOfFeatures[i]
        feature2=listOfFeatures[j]
        if abs(covarianceMatrix[feature1][feature2]) > 0.8: #If the correlation between the features is > 0.8
            setOfDroppedFeatures.add(feature1) #
print(setOfDroppedFeatures)
data = data.drop(setOfDroppedFeatures, axis=1)
nonCorrelatedWithOutput = [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.045]
#I tried different values of threshold and 0.045 was the one that gave the best results
print(nonCorrelatedWithOutput)
data = data.drop(nonCorrelatedWithOutput, axis=1)
newTrain = data.iloc[:1460]
newTest = data.iloc[1460:]
x= newTrain.drop("SalePrice", axis=1)

y= newTrain["SalePrice"]

model = LinearRegression() 
model = model.fit(x,y)

newTest = newTest.drop("SalePrice", axis=1)
pred = model.predict(newTest)
print(pred)
sub = pd.DataFrame() 
sub['Id'] = test['Id']
sub['SalePrice'] = pred
sub