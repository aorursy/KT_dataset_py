# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load the data train and test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#importing the liberary of linearregression

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor
#library for ploting the graphs

import matplotlib.pyplot as plt
#finding the numerical values which is usefull

train.head()

train.describe()
#cleaning the data

train["LotFrontage"].isnull().sum()

train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())
#claeaning the data

train["MasVnrArea"].isnull().sum()

train["MasVnrArea"] = train["MasVnrArea"].fillna(train["LotFrontage"].median())

train["MasVnrArea"].isnull().sum()
#cleanning the data

train["SalePrice"].isnull().sum()
#making the useful data set

new_columns=["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold"]

x_train = train[new_columns] 

y_train = train["SalePrice"]
train.columns
#checking the numeric data

train._get_numeric_data()
#calling the linear regression model

reg = linear_model.LinearRegression()


#fit the data to the model

reg.fit(x_train,y_train)
#checking the coffecient

reg.coef_
#now taking the test data

x_test = test[new_columns]

test.columns
#checking the numerical column

test.describe()
#cleaning the test data

test["LotFrontage"].isnull().sum()

test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].median())

test["LotFrontage"].isnull().sum()
#cleaning the test data

test["MasVnrArea"].isnull().sum()



test["MasVnrArea"] = test["MasVnrArea"].fillna(test["MasVnrArea"].median())

test["MasVnrArea"].isnull().sum()
#cleaning the test data

test["BsmtFinSF1"].isnull().sum()



test["BsmtFinSF1"] = test["BsmtFinSF1"].fillna(test["BsmtFinSF1"].median())

test["BsmtFinSF1"].isnull().sum()
#cleaning the test data

test["GarageArea"].isnull().sum()

test["GarageArea"] = test["GarageArea"].fillna(test["GarageArea"].median())

test["GarageArea"].isnull().sum()
#cleaning the test data

test["YrSold"].isnull().sum()


#now making the datframe of cleaned columns

x_test = test[new_columns]
#doing the linear predictions

a=reg.predict(x_test)


#value of the arrary (Prediction)

a
#intercept value

reg.intercept_
rf = RandomForestRegressor(n_estimators=3500,criterion='mse',max_leaf_nodes=3000,max_features='auto',oob_score=True)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
b=rf.predict(x_test)
id_col = test["Id"]

sale = ["SalePrice"]

newDf = pd.DataFrame({"SalePrice":sale} )

#submit = pd.DataFrame()

#submit = submit.append(id_col)

#submit = submit.append(a )
pd.to_numeric(submission["SalePrice"])

pd.to_numeric(submission["Id"])
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": b

    })



submission.to_csv('Submission.csv',index=False)
y_out = pd.read_csv("../input/sample_submission.csv")

y_test_out = np.array(y_out["SalePrice"])
#mean square error

sc = np.mean((a-y_test_out)**2)

#variance

score = reg.score(x_test,y_test_out)
#printing the mean square and variance

print(sc)

print(score)