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
sample_submission = pd.read_csv("../input/house-prices-dataset/sample_submission.csv")
test = pd.read_csv("../input/house-prices-dataset/test.csv")
train = pd.read_csv("../input/house-prices-dataset/train.csv")
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
test.head()
trainData = train.drop(columns=["Id"])
trainData.shape
test.shape
trainData.info()
trainData.describe(include = 'all')
corrs = trainData.corr()
plt.bar(trainData["OverallQual"], trainData["SalePrice"], label = 'Bar', align='center')
trainData = trainData.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])
trainData = trainData.drop(columns=["Exterior1st", "GarageYrBlt", "MSZoning", "HouseStyle", "Electrical", "GarageQual"])
trainData["LotFrontage"] = trainData["LotFrontage"].fillna(0)
trainData["MasVnrArea"] = trainData["MasVnrArea"].fillna(0)

trainData["GarageType"] = trainData["GarageType"].fillna("No Garage")
trainData["GarageCond"] = trainData["GarageCond"].fillna("No Garage")
trainData["GarageFinish"] = trainData["GarageFinish"].fillna("No Garage")
trainData["BsmtExposure"] = trainData["BsmtExposure"].fillna("No Basement")
trainData["BsmtFinType2"] = trainData["BsmtFinType2"].fillna("No Basement")
trainData["BsmtCond"] = trainData["BsmtCond"].fillna("No Basement")
trainData["BsmtQual"] = trainData["BsmtQual"].fillna("No Basement")
trainData["BsmtFinType1"] = trainData["BsmtFinType1"].fillna("No Basement")
trainData["MasVnrType"] = trainData["MasVnrType"].fillna("No Area")
obj_list = trainData.select_dtypes(include=["object"]).columns.to_list()
dummy = pd.get_dummies(trainData[obj_list])
dummy["SalePrice"] = trainData["SalePrice"]
X = dummy.iloc[:, dummy.columns != "SalePrice"].values
y = dummy.iloc[:, dummy.columns == "SalePrice"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)  
X_test = sc_x.transform(X_test)
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train)
len(X_test)
y_pred = classifier.predict(X_test)
len(y_pred)
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error, r2_score
print('mean_squared_error: ', mean_squared_error(y_test, y_pred),
     '\nr2_score: ',r2_score(y_test, y_pred)
     )
test.fillna(test.mean(), inplace=True)
test.head()
train["SalePrice"]
submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": y_pred
})

submission.to_csv('submission_output.csv', index=False)