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
# Import the Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read the Training and test set
data_train=pd.read_csv('../input/train.csv')
data_test=pd.read_csv('../input/test.csv')
#Concanat the datasets
data = pd.concat((data_train, data_test)).reset_index(drop=True)
#Basic Exploratory Data Analysis
data.head(5)
data.info()
#Data Analysis based on columns. Check for nulls and the types of data in non numeric columns
for datatype in data.columns:
    if ((data[datatype].dtype != np.float64) and (data[datatype].dtype != np.int64)):
        print("------------------------------------------------------")
        print("Name of Column", datatype)
        print("No of nulls ", data[datatype].isnull().value_counts())
        print("Value Count ", data[datatype].value_counts())
# Most of the non numeric columns have correct data. Now we need to remove nulls in the non numeric columns
# Column: Alley
# Since almost all of the values are missing in Alley, we will drop this column
data.drop(["Alley"],axis = 1, inplace = True)

# Column : Fence
data["Fence"].isnull().value_counts()
data["Fence"].value_counts()
# Since almost all of the values are missing in Fence, we will drop this column
data.drop(["Fence"],axis = 1, inplace = True)

# Column : MiscFeature
data["MiscFeature"].isnull().value_counts()
data["MiscFeature"].value_counts()
# Since almost all of the values are missing in MiscFeature, we will drop this column
data.drop(["MiscFeature"],axis = 1, inplace = True)

# Column : PoolQC
data["PoolQC"].isnull().value_counts()
data["PoolQC"].value_counts()
# Since almost all of the values are missing in PoolQC, we will drop this column
data.drop(["PoolQC"],axis = 1, inplace = True)

# Column : FireplaceQu
data["FireplaceQu"].isnull().value_counts()
data["FireplaceQu"].value_counts()
# Since almost half of the values are missing in FireplaceQu, we will drop this column
data.drop(["FireplaceQu"],axis = 1, inplace = True)

# Column : BsmtCond
data["BsmtCond"].isnull().value_counts()
data["BsmtCond"].value_counts()
# Since TA is the most common entry in this column, replace the nulls with TA
data["BsmtCond"].fillna(value= "TA",inplace=True)

# Column : BsmtExposure
data["BsmtExposure"].isnull().value_counts()
data["BsmtExposure"].value_counts()
# Since No is the most common entry in this column, replace the nulls with No
data["BsmtExposure"].fillna(value= "No",inplace=True)

# Column : BsmtFinType1
data["BsmtFinType1"].isnull().value_counts()
data["BsmtFinType1"].value_counts()
# Since Unf is the most common entry in this column, replace the nulls with Unf
data["BsmtFinType1"].fillna(value= "Unf",inplace=True)

# Column : BsmtFinType2
data["BsmtFinType2"].isnull().value_counts()
data["BsmtFinType2"].value_counts()
# Since Unf is the most common entry in this column, replace the nulls with Unf
data["BsmtFinType2"].fillna(value= "Unf",inplace=True)

# Column : BsmtQual
data["BsmtQual"].isnull().value_counts()
data["BsmtQual"].value_counts()
# Since TA is the most common entry in this column, replace the nulls with TA
data["BsmtQual"].fillna(value= "TA",inplace=True)

# Column : Electrical
data["Electrical"].isnull().value_counts()
data["Electrical"].value_counts()
# Since SBrkr is the most common entry in this column, replace the nulls with SBrkr
data["Electrical"].fillna(value= "SBrkr",inplace=True)

# Column : Exterior1st
data["Exterior1st"].isnull().value_counts()
data["Exterior1st"].value_counts()
# Since VinylSd is the most common value, replace nulls with VinylSd
data["Exterior1st"].fillna(value= "VinylSd",inplace=True)

# Column : Exterior2nd
data["Exterior2nd"].isnull().value_counts()
data["Exterior2nd"].value_counts()
# Since VinylSd is the most common value, replace nulls with VinylSd
data["Exterior2nd"].fillna(value= "VinylSd",inplace=True)

# Column : Functional
data["Functional"].isnull().value_counts()
data["Functional"].value_counts()
# Since Typ is the most common entry in this column, replace the nulls with Typ
data["Functional"].fillna(value= "Typ",inplace=True)

# Column : GarageCond
data["GarageCond"].isnull().value_counts()
data["GarageCond"].value_counts()
# Since TA is the most common entry in this column, replace the nulls with TA
data["GarageCond"].fillna(value= "TA",inplace=True)

# Column : GarageFinish
data["GarageFinish"].isnull().value_counts()
data["GarageFinish"].value_counts()
# Since Unf is the most common entry in this column, replace the nulls with Unf
data["GarageFinish"].fillna(value= "Unf",inplace=True)

# Column : GarageQual
data["GarageQual"].isnull().value_counts()
data["GarageQual"].value_counts()
# Since TA is the most common entry in this column, replace the nulls with TA
data["GarageQual"].fillna(value= "TA",inplace=True)

# Column : GarageType
data["GarageType"].isnull().value_counts()
data["GarageType"].value_counts()
# Since Attchd is the most common entry in this column, replace the nulls with Attchd
data["GarageType"].fillna(value= "Attchd",inplace=True)

# Column : KitchenQual
data["KitchenQual"].isnull().value_counts()
data["KitchenQual"].value_counts()
# Since TA is the most common entry in this column, replace the nulls with TA
data["KitchenQual"].fillna(value= "TA",inplace=True)

# Column : MSZoning
data["MSZoning"].isnull().value_counts()
data["MSZoning"].value_counts()
# Since RL is the most common value, replace nulls with RL
data["MSZoning"].fillna(value= "RL",inplace=True)

# Column : MasVnrType
data["MasVnrType"].isnull().value_counts()
data["MasVnrType"].value_counts()
# Since None is the most common value, replace nulls with None
data["MasVnrType"].fillna(value= "None",inplace=True)


# Column : SaleType
data["SaleType"].isnull().value_counts()
data["SaleType"].value_counts()
# Since WD is the most common entry in this column, replace the nulls with WD
data["SaleType"].fillna(value= "WD",inplace=True)

# Column : Utilities
data["Utilities"].isnull().value_counts()
data["Utilities"].value_counts()
# Since AllPub is the most common value, replace nulls with AllPub
data["Utilities"].fillna(value= "AllPub",inplace=True)

#Check for nulls and the types of data in numeric columns
for datatype in data.columns:
    if ((data[datatype].dtype != np.object)):
        print("------------------------------------------------------")
        print("Name of Column", datatype)
        print("No of nulls ", data[datatype].isnull().value_counts())
        print("count of values ", data[datatype].value_counts())

#Drop Column Id as it has no significance
data.drop(["Id"],axis = 1, inplace = True)
#Rearrange the columns so that Sale price is the last column
data = data[["1stFlrSF","2ndFlrSF","3SsnPorch","BedroomAbvGr","BldgType","BsmtCond","BsmtExposure","BsmtFinSF1","BsmtFinSF2","BsmtFinType1","BsmtFinType2","BsmtFullBath","BsmtHalfBath","BsmtQual","BsmtUnfSF","CentralAir","Condition1","Condition2","Electrical","EnclosedPorch","ExterCond","ExterQual","Exterior1st","Exterior2nd","Fireplaces","Foundation","FullBath","Functional","GarageArea","GarageCars","GarageCond","GarageFinish","GarageQual","GarageType","GarageYrBlt","GrLivArea","HalfBath","Heating","HeatingQC","HouseStyle","KitchenAbvGr","KitchenQual","LandContour","LandSlope","LotArea","LotConfig","LotFrontage","LotShape","LowQualFinSF","MSSubClass","MSZoning","MasVnrArea","MasVnrType","MiscVal","MoSold","Neighborhood","OpenPorchSF","OverallCond","OverallQual","PavedDrive","PoolArea","RoofMatl","RoofStyle","SaleCondition","SaleType","ScreenPorch","Street","TotRmsAbvGrd","TotalBsmtSF","Utilities","WoodDeckSF","YearBuilt","YearRemodAdd","YrSold","SalePrice"]]

#Create the features and target variable
X = data.iloc[:, :-1].values 
y = data.iloc[:, 74].values
#Fillin the missing values in numeric columns
from sklearn.preprocessing import Imputer

#Using Imputer for Null values in column "BsmtFinSF1"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,7:8])
X[:,7:8] = imputer.transform(X[:,7:8])

#Using Imputer for Null values in column "BsmtFinSF2"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,8:9])
X[:,8:9] = imputer.transform(X[:,8:9])

#Using Imputer for Null values in column "BsmtFullBath"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,11:12])
X[:,11:12] = imputer.transform(X[:,11:12])

#Using Imputer for Null values in column "BsmtHalfBath"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,12:13])
X[:,12:13] = imputer.transform(X[:,12:13])

#Using Imputer for Null values in column "BsmtUnfSF"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,14:15])
X[:,14:15] = imputer.transform(X[:,14:15])

#Using Imputer for Null values in column "GarageArea"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,28:29])
X[:,28:29] = imputer.transform(X[:,28:29])

#Using Imputer for Null values in column "GarageCars"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,29:30])
X[:,29:30] = imputer.transform(X[:,29:30])

#Using Imputer for Null values in column "GarageYrBlt"
imputer= Imputer(missing_values="NaN", strategy="median", axis=0)
imputer= imputer.fit(X[:,34:35])
X[:,34:35] = imputer.transform(X[:,34:35])

#Using Imputer for Null values in column "LotFrontage"
imputer= Imputer(missing_values="NaN", strategy="median", axis=0)
imputer= imputer.fit(X[:,46:47])
X[:,46:47] = imputer.transform(X[:,46:47])

#Using Imputer for Null values in column "MasVnrArea"
imputer= Imputer(missing_values="NaN", strategy="median", axis=0)
imputer= imputer.fit(X[:,51:52])
X[:,51:52] = imputer.transform(X[:,51:52])

#Using Imputer for Null values in column "TotalBsmtSF"
imputer= Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer= imputer.fit(X[:,68:69])
X[:,68:69] = imputer.transform(X[:,68:69])


#Convert categorical columns into numeric 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()

position =[4,5,6,9,10,13,15,16,17,18,20,21,22,23,25,27,30,31,32,33,37,38,39,41,42,43,45,47,50,52,55,59,61,62,63,64,66,69]
for i in position:
    X[:,i] = labelencoder_X.fit_transform(X[:,i])

col_names =["BldgType","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtQual","CentralAir","Condition1","Condition2","Electrical","ExterCond","ExterQual","Exterior1st","Exterior2nd","Foundation","Functional","GarageCond","GarageFinish","GarageQual","GarageType","Heating","HeatingQC","HouseStyle","KitchenQual","LandContour","LandSlope","LotConfig","LotShape","MSZoning","MasVnrType","Neighborhood","PavedDrive","RoofMatl","RoofStyle","SaleCondition","SaleType","Street","Utilities"]
for i in range(0,38):
    onehotencoder= OneHotEncoder(categorical_features=[i])
    X=onehotencoder.fit_transform(X).toarray()
# Since both the test and train data has been combined, they need to be removed seperately
X_data_train = X[:1460,:]
X_data_test  = X[1460:,:]
y = y[:1460]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data_train, y, test_size = 0.2, random_state = 0)
#------------------Prediction ------------------------

#-- Multiple Linear Regression
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Vusualizing the results
plt.scatter(y_test,y_pred)
plt.show()
#Regression result Evaluation to generate below co-efficients
from sklearn import metrics
#Mean Absolute Error (MAE)
metrics.mean_absolute_error(y_test,y_pred)
#Mean Squared Error(MSE)
metrics.mean_squared_error(y_test,y_pred)
#Root Mean Squared Error(RMSE)
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
#Predicting results using Random Forest
# -----------------Random Forest Regression-----------------------
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred_rf = regressor.predict(X_test)


# Vusualizing the results
plt.scatter(y_test,y_pred_rf)
plt.show()
sns.distplot(y_test-y_pred_rf, bins = 50)
#Mean Absolute Error (MAE)
metrics.mean_absolute_error(y_test,y_pred_rf)
#Mean Squared Error(MSE)
metrics.mean_squared_error(y_test,y_pred_rf)
#Root Mean Squared Error(RMSE)
np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf))
# Since ramdom forest has given the best results, the predictions for the test file 
# would be done based on this model only
#Decision Tree predictions for test file
y_pred_test_rf = regressor.predict(X_data_test)
#Create submission file
price_submission = pd.DataFrame({'Id': data_test.Id, 'SalePrice': y_pred_test_rf})
# you could use any filename. We choose submission here
price_submission.to_csv('house_price_submission.csv', index=False)
