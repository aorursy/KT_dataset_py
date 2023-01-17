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
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,recall_score,precision_score



data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

print(data.columns)

print(data.head(5))
# checking for null values in the columns

columns_with_missing_values = [col for col in data.columns if data[col].isnull().any()]

print(columns_with_missing_values)
#countplot on dependable variable to understand the variance 

sns.countplot("target",data=data)
# pairplot between the independent variables to check correlation 

sns.pairplot(data,kind="reg")
# splitting data for training and testing

independent_variables = data.iloc[:,0:13]

dependent_variable = data.iloc[:,13:14]



xtrain,xtest,ytrain,ytest = train_test_split(independent_variables,dependent_variable,test_size=0.25)



#initializing model

logisticReg = LogisticRegression()

logisticReg.fit(xtrain,ytrain)

#prediction 

ypred = logisticReg.predict(xtest)



#metrics 



result = mean_squared_error(ytest,ypred)

print("mean_squared_error",result)

abresult = mean_absolute_error(ytest,ypred)

print("mean_absolute_error",abresult)

accuracyResult=accuracy_score(ytest,ypred)

print("accuracy_score",accuracyResult*100)

recallResult = recall_score(ytest,ypred)

print("recall_score",recallResult*100)

precisionResult = precision_score(ytest,ypred)

print("precision_score",precisionResult*100)








