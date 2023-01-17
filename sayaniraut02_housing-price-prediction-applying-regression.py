# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings('ignore')

import scipy.stats as st

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train.head(10)
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test.head()
train.columns
train.isnull().mean()
test.isnull().mean()
train.shape
train.describe()
test.describe()
def impute_na_numeric(train,test,var):

    mean = train[var].mean()

    median = train[var].median()

    

    train[var+"_mean"] = train[var].fillna(mean)

    train[var+"_median"] = train[var].fillna(median)

    

    var_original = train[var].std()**2

    var_mean = train[var+"_mean"].std()**2

    var_median = train[var+"_median"].std()**2

    

    print("Original Variance: ",var_original)

    print("Mean Variance: ",var_mean)

    print("Median Variance: ",var_median)

    

    if((var_mean < var_original) | (var_median < var_original)):

        if(var_mean < var_median):

            train[var] = train[var+"_mean"]

            test[var] = test[var].fillna(mean)

        else:

            train[var] = train[var+"_median"]

            test[var] = test[var].fillna(median)

    else:

        test[var] = test[var].fillna(median)

    train.drop([var+"_mean",var+"_median"], axis=1, inplace=True)

            
na_num=['LotFrontage','BsmtFullBath','BsmtHalfBath','GarageArea','GarageCars','TotalBsmtSF','MasVnrArea']
for i in na_num:

    print(i)

    impute_na_numeric(train,test,i)

    print(" ")
train["GarageYrBlt"].mode().values[0]
def impute_na_non_numeric(train,test,var):

    mode = train[var].mode().values[0]

    train[var] = train[var].fillna(mode)

    test[var] = test[var].fillna(mode)
na_cat=['Electrical','GarageQual','FireplaceQu','SaleType','Exterior1st','Exterior2nd','Utilities','BsmtFinType1','BsmtFinSF1','Heating','FullBath','HalfBath','Functional','GarageType','GarageYrBlt','GarageFinish','Condition1','Condition2','MasVnrType','MiscFeature','BsmtFullBath','BsmtHalfBath']
for i in na_cat:

    impute_na_non_numeric(train,test,i)
from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(train[c].values)) 

    train[c] = lbl.transform(list(train[c].values))

    lbl.fit(list(test[c].values)) 

    test[c] = lbl.transform(list(test[c].values))

    
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

train['Total_sqr_footage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])

train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))



train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])

test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))



test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])
drop_cols = ['Id','TotalBsmtSF','1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF','LandSlope','Utilities','LandContour','LandSlope','MasVnrType']
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, annot=True);
# most correlated features

corrmat = train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.barplot(train.OverallQual,train.SalePrice)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
plt.scatter(train.GrLivArea, train.SalePrice, c = "red", marker = "s")

plt.title("Predicting outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()



train = train[train.GrLivArea < 4000]
plt.scatter(train.GrLivArea, train.SalePrice, c = "red", marker = "s")

plt.title("Predicting outliers")

plt.xlabel("GarageArea")

plt.ylabel("SalePrice")

plt.show()



train = train[train['GarageArea'] < 1200]
y = train['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
# Log transform the target variable

train.SalePrice = np.log1p(train.SalePrice)

y = train.SalePrice
plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
train.drop(drop_cols,axis=1).drop(["SalePrice"],axis=1).columns
x_t=train.drop(drop_cols,axis=1).drop(["SalePrice"],axis=1)

x_t
y_t = train['SalePrice']
# to convert a categorical into one hot encoding

hot_one= pd.get_dummies(x_t)

test=pd.get_dummies(test)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train ,y_test = train_test_split(hot_one, y_t, test_size = 0.3, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)

x_test = sc_X.transform(x_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,SGDRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,VotingRegressor
regression_models = ['LinearRegression','ElasticNet','Lasso','Ridge','SGDRegressor','SVR',

                    'DecisionTreeRegressor','RandomForestRegressor','AdaBoostRegressor']
mse = []

rmse = []

mae = []

models = []

estimators = []
for reg_model in regression_models:

    

    model = eval(reg_model)()

    

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    

    models.append(type(model).__name__)

    estimators.append((type(model).__name__,model))

    

    mse.append(mean_squared_error(y_test,y_pred))

    rmse.append(mean_squared_error(y_test,y_pred)**0.5)

    mae.append(mean_absolute_error(y_test,y_pred))
model_dict = {"Models":models,

             "MSE":mse,

             "RMSE":rmse,

             "MAE":mae}
model_df = pd.DataFrame(model_dict)

model_df
model_df["Inverse_Weights"] = model_df['RMSE'].map(lambda x: np.log(1.0/x))

model_df
vr = VotingRegressor(estimators=estimators,weights=model_df.Inverse_Weights.values)
vr.fit(x_train,y_train)
y_pred = vr.predict(x_test)
models.append("Voting_Regressor")

mse.append(mean_squared_error(y_test,y_pred))

rmse.append(mean_squared_error(y_test,y_pred)**0.5)

mae.append(mean_absolute_error(y_test,y_pred))
sub_df = pd.concat([test['Id'],

                    pd.DataFrame(y_pred,columns=["SalePrice"])],

                   axis=1)

sub_df.head()
sub_df.to_csv("Stacked_Ensemble_Baseline_Submission.csv", index=False)