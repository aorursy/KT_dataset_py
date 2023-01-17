import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn 

import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.describe()



trnID = train['Id']

tstID = test['Id'] #in case we need this column later, but we probably won't



train.drop("Id",axis = 1,inplace = True)

test.drop("Id",axis = 1,inplace = True)

#handle missing data

whole_data = pd.concat((train, test)).reset_index(drop=True)

whole_data.drop('SalePrice',axis = 1,inplace = True) # the test data does not have this column

data_missing = (whole_data.isnull().sum()/len(whole_data)).sort_values(ascending = False)

missing_data_ratio = pd.DataFrame({'MissingRatio':data_missing})



missing_data_ratio.head(40)

def fill_missing (feat,fillstr):

    train[feat]=train[feat].fillna(fillstr)

    test[feat]=test[feat].fillna(fillstr)



#PoolQC NA = No Pool (from datadescription)

fill_missing("PoolQC","None")

#MiscFeature NA = No Misc Features (from datadescription)

fill_missing("MiscFeature","None")

#Alley NA = "no alley access" (from datadescription)

fill_missing("Alley","None")

#Fence NA = "No fence" (from datadescription)

fill_missing("Fence","None")

#FireplaceQu NA = no fireplace

fill_missing("FireplaceQu","None")

#LotFrontage weird to have 16% of these measures missing, but most likely a house's lot frontage will be similar to that of houses near it.

#Therefore, we can average the lot frontage of houses in its neightborhood, and use that value in each case.

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.mean())) #found this method from the Serigne notebook, although that user opted to use the median

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.mean()))

#fill the rest

fill_missing("GarageType","None")

fill_missing("GarageFinish","None")

fill_missing("GarageQual","None")

fill_missing("GarageCond","None")

fill_missing("GarageYrBlt",0) #numerical data gets filled with 0s

fill_missing("GarageArea", 0)

fill_missing("GarageCars", 0)

fill_missing("BsmtQual","None")

fill_missing("BsmtExposure","None")

fill_missing("BsmtQual","None")

fill_missing("BsmtFinType1","None")

fill_missing("BsmtFinType2","None")

fill_missing("BsmtFinSF1",0)

fill_missing("BsmtFinSF2",0)

fill_missing("BsmtUnfSF",0)

fill_missing("BsmtFullBath",0)

fill_missing("BsmtHalfBath",0)

fill_missing("MasVnrType","None")

fill_missing("MasVnrArea",0)

fill_missing("MSZoning",whole_data["MSZoning"].mode())#data description doesnt cover the default for this non-numerical variable, so I used the most common value.

fill_missing("Electrical",whole_data["Electrical"].mode()) #similarly

fill_missing("KitchenQual",whole_data["KitchenQual"].mode())  

fill_missing("Exterior1st",whole_data["Exterior1st"].mode())  

fill_missing("Exterior2nd",whole_data["Exterior2nd"].mode())  

fill_missing("SaleType",whole_data["SaleType"].mode())

fill_missing("Functional","Typical") #from data description

#All the notebooks I read decided to drop or ignore the Utilities feature, since it has the same value for all but 3 of the observations. Instead of taking the most common value, I will just follow suit.

train.drop(["Utilities"],axis = 1)

test.drop(["Utilities"],axis = 1)

#remove outliers

plt.scatter(train.GrLivArea, train.SalePrice, c = "black", marker = "s")

plt.title("Outlier Detection")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
train = train[train.GrLivArea < 4000]
for col in train.columns:

    if train[col].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[col].values))

        train[col]= lbl.transform(list(train[col].values))

train.describe()

corr= train.corr()

corr.sort_values(["SalePrice"],ascending= False,inplace = True)

print(corr.SalePrice)
feats = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","FullBath","TotRmsAbvGrd","YearBuilt","YearRemodAdd"]

X = train[feats]

y = train["SalePrice"]


rf = RandomForestRegressor()

lr = LinearRegression()

xr = xgb.XGBRegressor()

def mscore(model,mname):

    model.fit(X,y)

    scores = cross_val_score(rf,X,y,scoring = "neg_mean_squared_error", cv = 10)

    rmse_score = np.sqrt(-scores) #this is from the Geron book



    print("\n Scores for:", mname)

    print("\n Mean: ", rmse_score.mean() )

    print("\n Standard Deviation: ", rmse_score.std() )

mscore(rf, "Random Forest")

mscore(lr, "Linear Regression")

mscore(xr, "XGBoost Regressor")
submsn = pd.DataFrame()

submsn['Id'] = tstID

output = lr.predict(test[feats].fillna(0))

submsn['SalePrice']= output

submsn.shape

submsn.to_csv('submission.csv',index = False)




