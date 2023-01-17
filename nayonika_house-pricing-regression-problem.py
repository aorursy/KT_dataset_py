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
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_df.head(10)
train_df.info()
test_df.info()
data = pd.concat((train_df, test_df)).reset_index(drop=True)

x_saleprice = train_df["SalePrice"]

data.drop(["SalePrice"], axis = 1, inplace= True)

data.shape
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)

import seaborn as sns

plt.figure(figsize = (15,5))



plt.title("Heatmap to see the nulls")

sns.heatmap(data.isnull(), yticklabels = False, cbar = False)

plt.xlabel("Features")

plt.ylabel("Nulls as white lines")
x = data.isnull().sum().sort_values(ascending = False)

print (x[x>0])
#Filling blank values for the following features with NA

data["PoolQC"] = data["PoolQC"].fillna("NA")

data["MiscFeature"] = data["MiscFeature"].fillna("NA")

data["Alley"] = data["Alley"].fillna("NA")

data["FireplaceQu"] = data["FireplaceQu"].fillna("NA")

data["Fence"] = data["Fence"].fillna("NA")

data["GarageQual"] = data["GarageQual"].fillna("NA")

data["GarageYrBlt"] = data["GarageYrBlt"].fillna("NA")

data["GarageCond"] = data["GarageCond"].fillna("NA")

data["GarageFinish"] = data["GarageFinish"].fillna("NA")

data["GarageType"] = data["GarageType"].fillna("NA")

data["BsmtCond"] = data["BsmtCond"].fillna("NA")

data["BsmtExposure"] = data["BsmtExposure"].fillna("NA")

data["BsmtQual"] = data["BsmtQual"].fillna("NA")

data["BsmtFinType2"] = data["BsmtFinType2"].fillna("NA")

data["BsmtFinType1"] = data["BsmtFinType1"].fillna("NA")

data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())

data["MSZoning"] = data["MSZoning"].fillna(data["MSZoning"].mode()[0])

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

data["Functional"] = data["Functional"].fillna(data["Functional"].mode()[0])

data["Utilities"] = data["Utilities"].fillna(data["Utilities"].mode()[0])

data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(data["BsmtHalfBath"].mode()[0])

data["BsmtFullBath"] = data["BsmtFullBath"].fillna(data["BsmtFullBath"].mode()[0])

data["BsmtFinSF2"] = data["BsmtFinSF2"].fillna(data["BsmtFinSF2"].mode()[0])

data["BsmtFinSF1"] = data["BsmtFinSF1"].fillna(data["BsmtFinSF1"].mode()[0])

data["GarageArea"] = data["GarageArea"].fillna(data["GarageArea"].mode()[0])

data["Exterior1st"] = data["Exterior1st"].fillna(data["Exterior1st"].mode()[0])

data["BsmtUnfSF"] = data["BsmtUnfSF"].fillna(data["BsmtUnfSF"].mode()[0])

data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(data["TotalBsmtSF"].mode()[0])

data["GarageCars"] = data["GarageCars"].fillna(data["GarageCars"].mode()[0])

data["Exterior2nd"] = data["Exterior2nd"].fillna(data["Exterior2nd"].mode()[0])

data["KitchenQual"] = data["KitchenQual"].fillna(data["KitchenQual"].mode()[0])

data["SaleType"] = data["SaleType"].fillna(data["SaleType"].mode()[0])

data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])
objList = data.select_dtypes(include = "object").columns



print (objList)
def one_hot(df, cols):

    """

    @param df pandas DataFrame

    @param cols a list of columns to encode 

    @return a DataFrame with one-hot encoding

    """

    i = 0

    for each in cols:

        #print (each)

        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)

        if i == 0: 

            print (dummies)

            i = i + 1

        df = pd.concat([df, dummies], axis=1)

    return df
#One hot encoding done

data = one_hot(data, objList) 

data.shape
#Dropping duplicates columns if any

data = data.loc[:,~data.columns.duplicated()]

data.shape
#Dropping the original columns that has data type object 

data.drop(objList, axis=1, inplace=True)

data.shape
train_df = data.iloc[:1460,:]

test_df = data.iloc[1460 :,:]
train_df["SalePrice"] = x_saleprice
X_train = train_df.drop(["SalePrice"], axis = 1)

Y_train = train_df["SalePrice"]

X_test = test_df
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error



n_folds = 5



cv = KFold(n_splits = 5, shuffle=True, random_state=42).get_n_splits(X_train.values)



def test_model(model):   

    msle = make_scorer(mean_squared_log_error)

    rmsle = np.sqrt(cross_val_score(model, X_train, Y_train, cv=cv, scoring = msle))

    score_rmsle = [rmsle.mean()]

    return score_rmsle



def test_model_r2(model):

    r2 = make_scorer(r2_score)

    r2_error = cross_val_score(model, X_train, Y_train, cv=cv, scoring = r2)

    score_r2 = [r2_error.mean()]

    return score_r2
from sklearn.ensemble import GradientBoostingRegressor

clf_ggr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',

                          init=None, learning_rate=0.1, loss='ls', max_depth=3,

                          max_features='sqrt', max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=1, min_samples_split=2,

                          min_weight_fraction_leaf=0.0, n_estimators=1750,

                          n_iter_no_change=None, presort='deprecated',

                          random_state=None, subsample=0.85, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)



rmsle_ggr = test_model(clf_ggr)

print (rmsle_ggr, test_model_r2(clf_ggr))

#[0.13238059479479883] [0.8856026444419683]

#[0.13948744569029933] [0.8818334439186121]
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



param_grid = {'min_samples_split':[2,4,6,8,10,20,40,60,100], 

              'min_samples_leaf':[1,3,5,7,9, 15, 20, 25, 30, 40, 50],

              'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1],

              'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001, 0.2], 

              'n_estimators':[10, 30, 50, 100,250,500,750,1000,1250,1500,1750],

              'max_features' : ['sqrt']

             }



asdf = GradientBoostingRegressor()



#clf = GridSearchCV(asdf, param_grid=param_grid, scoring='r2', n_jobs=-1)

clf = RandomizedSearchCV(asdf, param_grid, scoring='r2', n_jobs=-1)

 

clf.fit(X_train, Y_train)



print(clf.best_estimator_)
#Training model



clf_ggr.fit(X_train, Y_train)

Y_pred = clf_ggr.predict(test_df) 
pred=pd.DataFrame(Y_pred)

sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)

print("Your submission was successfully saved!")