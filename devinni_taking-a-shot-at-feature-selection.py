# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



train = pd.read_csv("../input/train.csv", index_col="Id")



#print all numeric columns of the dataset

train_numerics = train._get_numeric_data()

print(train_numerics.columns)


features_linear = ["OverallQual","YearBuilt", "YearRemodAdd","GarageYrBlt",]

features_linear_log = ["LotFrontage", "LotArea", "1stFlrSF",]

features_nonlinear_log = ["BsmtFinSF1","TotalBsmtSF","MasVnrArea", "BsmtFinSF1",]



features_excludezeros = ["2ndFlrSF","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","ScreenPorch","MiscVal",]

#for these features: exclude the non-zero values -> e.g. for 2ndFlrSF these 0-values represent houses without a second floor

features_noclearrelationshipfoundyet = ["MSSubClass","OverallCond","BsmtFinSF2","BsmtUnfSF","LowQualFinSF","BsmtFullBath","BsmtHalfBath","FullBath",

                                        "HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","35snPorch","PoolArea","MoSold","YrSold","GRLivArea",]



#for myfeature in train_numerics.columns:

for myfeature in features_linear:

    plt.figure(figsize=(16,6))

    sns.regplot(x=train[myfeature],y=train["SalePrice"])

    

for myfeature in features_linear_log:

    plt.figure(figsize=(16,6))

    sns.regplot(x=np.log(train[myfeature]),y=train["SalePrice"])
train_X_1 = pd.concat([train[features_linear],np.log(train[features_linear_log])], axis=1)



#train_1.head()

train_X_1.describe()



#We replace all NaN values with the mean

train_X_1_NaNremoved = train_X_1.fillna(train_X_1.mean())



train_X_1_NaNremoved.describe()
def Preprocessing_1(X,features_linear,features_linear_log):

    #(1) We only keep the relevant features for this model

    X_features_selected = pd.concat([X[features_linear],np.log(X[features_linear_log])], axis=1)

    

    #(2)We replace all NaN values with the mean

    X_NaNremoved = X_features_selected.fillna(X_features_selected.mean())

    

    return X_NaNremoved

    

    
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X = train.drop("SalePrice",axis=1)

y = train["SalePrice"]



train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)



train_X = Preprocessing_1(train_X,features_linear,features_linear_log)

val_X = Preprocessing_1(val_X,features_linear,features_linear_log)



#Fit the model to the test dataset

myLinearRegressionModel = LinearRegression()

myLinearRegressionModel.fit(train_X, train_y)



#predict the values on the validation dataset

pred_val_y = myLinearRegressionModel.predict(val_X)



#correct negative predictions... Not clear yet why this occurs -> to be clarified

pred_val_y = np.where(pred_val_y < 0, pred_val_y.mean(), pred_val_y)



RMSE = np.sqrt(mean_squared_error(np.log(val_y), np.log(pred_val_y)))



print("RMSE = " + str(RMSE))
from sklearn.ensemble import RandomForestRegressor



myRandomForestModel = RandomForestRegressor(random_state=1)

myRandomForestModel.fit(train_X, train_y)



#predict the values on the validation dataset

pred_val_y = myRandomForestModel.predict(val_X)



RMSE = np.sqrt(mean_squared_error(np.log(val_y), np.log(pred_val_y)))



print("RMSE = " + str(RMSE))
X = train.drop("SalePrice",axis=1)

y = train["SalePrice"]



train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

#preprocessing (simple feature selection in this example)

train_X = train_X[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']]

val_X = val_X[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']]

myRandomForestModel.fit(train_X, train_y)



#predict the values on the validation dataset

pred_val_y = myRandomForestModel.predict(val_X)



RMSE = np.sqrt(mean_squared_error(np.log(val_y), np.log(pred_val_y)))



print("RMSE = " + str(RMSE))
X = train.drop("SalePrice",axis=1)

y = train["SalePrice"]



train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)



features_linear = ["OverallQual","YearBuilt", "YearRemodAdd","GarageYrBlt","TotRmsAbvGrd"]

features_linear_log = ["LotFrontage", "LotArea", "1stFlrSF",]

train_X = Preprocessing_1(train_X,features_linear,features_linear_log)

val_X = Preprocessing_1(val_X,features_linear,features_linear_log)



myRandomForestModel = RandomForestRegressor(random_state=1)

myRandomForestModel.fit(train_X, train_y)



#predict the values on the validation dataset

pred_val_y = myRandomForestModel.predict(val_X)



RMSE = np.sqrt(mean_squared_error(np.log(val_y), np.log(pred_val_y)))



print("RMSE = " + str(RMSE))
plt.figure(figsize=(16,6))

sns.regplot(x=train["TotRmsAbvGrd"],y=train["SalePrice"])
#Train on full training dataset

train_X = X

train_y = y

train_X = Preprocessing_1(train_X,features_linear,features_linear_log)

myRandomForestModel.fit(train_X, train_y)



#

test_X = pd.read_csv("../input/test.csv", index_col="Id")

test_X = Preprocessing_1(test_X, features_linear, features_linear_log)



pred_test_y = myRandomForestModel.predict(test_X)



#correct negative predictions... Not clear yet why this occurs -> to be clarified

pred_test_y = np.where(pred_test_y < 0, pred_val_y.mean(), pred_test_y)



print(pred_test_y.tolist())



my_submission = pd.DataFrame({'Id': test_X.index, 'SalePrice': pred_test_y})

my_submission.to_csv('submission.csv', index=False)