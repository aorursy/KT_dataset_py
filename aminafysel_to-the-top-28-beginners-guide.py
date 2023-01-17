# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(os.listdir("../input"))
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()
train_df.shape
test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.head()
train_df.info()
train_df.describe()
test_df.describe()
all_data=pd.concat((train_df,test_df)).reset_index(drop=True)

x_saleprice=train_df["SalePrice"]

all_data.drop(["SalePrice"],axis=1,inplace=True)

all_data.shape
from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(train_df['SalePrice'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
# log(1+x) transform

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 

sns.distplot(train_df['SalePrice'] , fit=norm, color="b");



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_df['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)



plt.show()
# Remove outliers

train_df.drop(train_df[(train_df['OverallQual']<5) & (train_df['SalePrice']>200000)].index, inplace=True)

train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<300000)].index, inplace=True)

train_df.reset_index(drop=True, inplace=True)
sns.heatmap(all_data.isnull(),yticklabels=False,cbar=False)
all_data['LotFrontage']=all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean())

all_data.drop(['Alley'],inplace=True,axis=1)

all_data['BsmtQual']=all_data['BsmtQual'].fillna(all_data['BsmtQual'].mode()[0])

all_data['BsmtCond']=all_data['BsmtCond'].fillna(all_data['BsmtCond'].mode()[0])

all_data['FireplaceQu']=all_data['FireplaceQu'].fillna(all_data['FireplaceQu'].mode()[0])

all_data['GarageType']=all_data['GarageType'].fillna(all_data['GarageType'].mode()[0])

all_data['GarageQual']=all_data['GarageQual'].fillna(all_data['GarageQual'].mode()[0])

all_data['GarageCond']=all_data['GarageCond'].fillna(all_data['GarageCond'].mode()[0])



all_data.drop(['PoolQC','Fence','MiscFeature','Id'],inplace=True,axis=1)

all_data['MasVnrType']=all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])

all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mode()[0])

all_data['BsmtExposure']=all_data['BsmtExposure'].fillna(all_data['BsmtExposure'].mode()[0])

all_data['BsmtFinType1']=all_data['BsmtFinType1'].fillna(all_data['BsmtFinType1'].mode()[0])

all_data['BsmtFinType2']=all_data['BsmtFinType2'].fillna(all_data['BsmtFinType2'].mode()[0])

all_data.drop(['GarageYrBlt'],inplace=True,axis=1)

all_data['GarageFinish']=all_data['GarageFinish'].fillna(all_data['GarageFinish'].mode()[0])

all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])















                                                
all_data['Utilities']=all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])

all_data['Exterior1st']=all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd']=all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['BsmtFinType1']=all_data['BsmtFinType1'].fillna(all_data['BsmtFinType1'].mode()[0])

all_data['BsmtFinSF1']=all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].mean())

all_data['BsmtFinSF2']=all_data['BsmtFinSF2'].fillna(all_data['BsmtFinSF2'].mean())

all_data['BsmtUnfSF']=all_data['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].mean())

all_data['TotalBsmtSF']=all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].mean())

all_data['BsmtFullBath']=all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mode()[0])

all_data['BsmtHalfBath']=all_data['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mode()[0])

all_data['KitchenQual']=all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Functional']=all_data['Functional'].fillna(all_data['Functional'].mode()[0])

all_data['GarageCars']=all_data['GarageCars'].fillna(all_data['GarageCars'].mean())

all_data['GarageArea']=all_data['GarageArea'].fillna(all_data['GarageArea'].mean())

all_data['SaleType']=all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

sns.heatmap(all_data.isnull(),yticklabels=False,cbar=False)


categorical_feature_mask = all_data.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = all_data.columns[categorical_feature_mask].tolist()
categorical_cols
len(categorical_cols)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

all_data[categorical_cols] = all_data[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
all_data.shape
all_data.head()
train_df = all_data.iloc[:1460,:]  

test_df = all_data.iloc[1460 :,:]  
train_df["SalePrice"] = x_saleprice
train_df.info()
X_train=train_df.drop(['SalePrice'],axis=1)

y_train=train_df['SalePrice']

X_test=test_df
X_train.shape
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error



n_folds = 5



cv = KFold(n_splits = 5, shuffle=True, random_state=42).get_n_splits(X_train.values)



def test_model(model):   

    msle = make_scorer(mean_squared_log_error)

    rmsle = np.sqrt(cross_val_score(model, X_train, y_train, cv=cv, scoring = msle))

    score_rmsle = [rmsle.mean()]

    return score_rmsle



def test_model_r2(model):

    r2 = make_scorer(r2_score)

    r2_error = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)

    score_r2 = [r2_error.mean()]

    return score_r2
from sklearn.ensemble import GradientBoostingRegressor

reg_gbr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',

                          init=None, learning_rate=0.05, loss='ls', max_depth=3,

                          max_features='sqrt', max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=9, min_samples_split=8,

                          min_weight_fraction_leaf=0.0, n_estimators=1250,

                          n_iter_no_change=None, presort='deprecated',

                          random_state=None, subsample=0.8, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)



rmsle_ggr = test_model(reg_gbr)

print (rmsle_ggr, test_model_r2(reg_gbr))
reg_gbr.fit(X_train, y_train)

y_pred  = reg_gbr.predict(test_df) 
y_pred
y_pred = pd.DataFrame(y_pred, columns=['SalePrice'])
y_pred.head()
y_pred.tail()
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

results=pd.concat([sub_df['Id'],y_pred],axis=1)
results.columns=['Id','SalePrice']

results.to_csv('submission.csv',index=False)

print("Your submission was successfully saved!")
