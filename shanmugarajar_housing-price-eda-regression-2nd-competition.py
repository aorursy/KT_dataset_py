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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from scipy import stats
df1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df2 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df1.head()
#Let us check the missing values and correlation between the features, if possible

ax, f = plt.subplots(figsize=(12,15))

sns.heatmap(df1.corr(), square=True)
#This is challenging to infer(atleast for me), so going to try manual check for correlation value >0.7

df1.corr().to_csv('t1.csv')

df2.corr().to_csv('t2.csv')
#Let us get the missing values now

miss_col_train=df1.columns[df1.isna().any()].tolist()

miss_col_test=df2.columns[df2.isna().any()].tolist()
for i in miss_col_train:

    miss_perc1=100-(df1[i].count()/len(df1[i]))*100

    print(i,"  missing percentage values %2.2f "%miss_perc1)

print('\n********************************\n')

for j in miss_col_test:

    miss_perc2=100-(df2[j].count()/len(df2[j]))*100

    print(j,"  missing percentage values %2.2f "%miss_perc2)
#On checking the data description, leaving few features, rest missing can be inferred as Not Applicable/ Not Available. 

#For eg. missing values in Alley cabn be rightly interpreted as No Alley and not as None. So we group such features and

#replace their missing values as appropriate

new_miss_column = ['Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

                   'BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','FireplaceQu',

                   'GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageQual','GarageCond',

                   'PoolQC','Fence','MiscFeature']



for i in new_miss_column:

    df1[i].fillna('Not Applicable', inplace=True)

    df2[i].fillna('Not Applicable', inplace=True)
#Let us re-check the missing columns again

miss_col_train=df1.columns[df1.isna().any()].tolist()

miss_col_test=df2.columns[df2.isna().any()].tolist()

print(miss_col_train)

print(miss_col_test)
#The left over columns are not items that are Not Applicable, so let us make them as most common occurence if the feature is categorical

obj_col_train=[]

obj_col_test=[]

for i in miss_col_train:

    if df1[i].dtype=='object':

        obj_col_train.append(i)

        miss_col_train.remove(i)



print('\n***********************\n')  



for i in miss_col_test:

    if df2[i].dtype=='object':

        obj_col_test.append(i)

        miss_col_test.remove(i)

        



print(obj_col_train, obj_col_test)

print('Non object columns  ', miss_col_train, miss_col_test)
#Replacing the categorical columns with most common occurence

for i in obj_col_train:

    df1[i].fillna(df1[i].mode()[0], inplace=True)



for i in obj_col_test:

    df2[i].fillna(df2[i].mode()[0], inplace=True)
#Now proceeding for Numerical columns - LotFRontage

fig,(ax1,ax2) = plt.subplots(ncols=2)

sns.boxplot(df1['LotFrontage'],ax=ax1)

sns.boxplot(df2['LotFrontage'],ax=ax2)
#There are outliers, let us not rmeove them, rather check the median for non-outliers (like 125) and replace for missing values

df1['LotFrontage'].fillna(df1[df1['LotFrontage']<125]['LotFrontage'].median(),inplace=True)

df2['LotFrontage'].fillna(df2[df2['LotFrontage']<125]['LotFrontage'].median(),inplace=True)
#Check the distribution of LotFrontage after change

sns.distplot(df1['LotFrontage'])

fig = plt.figure()

res = stats.probplot(df1['LotFrontage'], plot=plt)
#Check the distribution of LotFrontage in test

sns.distplot(df2['LotFrontage'])

fig = plt.figure()

res = stats.probplot(df2['LotFrontage'], plot=plt)
## It is observed that they are not normally ditrbuted and checking through other notebooks, we standardise them using log

df1['LotFrontage'] = np.log(df1['LotFrontage'])

df2['LotFrontage'] = np.log(df2['LotFrontage'])
#Re-check the distribution

sns.distplot(df1['LotFrontage'])

fig = plt.figure()

res = stats.probplot(df1['LotFrontage'], plot=plt)
sns.distplot(df2['LotFrontage'])

fig = plt.figure()

res = stats.probplot(df2['LotFrontage'], plot=plt)
#Now the data is normalised, let us look into  Electrical, Utlilities, MSZoning,'Exterior1st', 'KitchenQual' all are string 

#and 'GarageArea' numerical.Let us fill the missing values for them - Garage let us make it 0



df1['Electrical'].fillna(df1['Electrical'].mode()[0], inplace=True)

df2['Utilities'].fillna(df2['Utilities'].mode()[0], inplace=True)

df2['MSZoning'].fillna(df2['MSZoning'].mode()[0], inplace=True)

df2['Exterior1st'].fillna(df2['Exterior1st'].mode()[0], inplace=True)

df2['KitchenQual'].fillna(df2['KitchenQual'].mode()[0], inplace=True)

df2['GarageArea'].fillna(0, inplace=True)
#Re-check missing values

print(df1.columns[df1.isna().any()])

print(df2.columns[df2.isna().any()])
#Finally let us check the target fearture (SalePrice) and its disrtibution

sns.distplot(df1['SalePrice'])

fig=plt.figure()

stats.probplot(df1['SalePrice'], plot=plt)
#As expected they are not normally distributed. We have to log transform them.

df1['SalePrice'] = np.log(df1['SalePrice'])
#Re-check the distribution now

sns.distplot(df1['SalePrice'])

fig=plt.figure()

stats.probplot(df1['SalePrice'], plot=plt)
#We shall proceed with simple categorical conversion using code to convert categories to numerical

cat_cols_train = df1.select_dtypes(include=np.object)

cat_cols_test = df2.select_dtypes(include=np.object)



for i in cat_cols_train:

    df1[i] = df1[i].astype('category')

    df1[i] = df1[i].cat.codes



for j in cat_cols_test:

    df2[j] = df2[j].astype('category')

    df2[j] = df2[j].cat.codes
#Now all changes are done we shall check the contents and also see for any improved collinearity between features

plt.figure(figsize=(15,12))

sns.heatmap(df1.corr(), square=True)
#Safe keeping of the data and create new working data frame for train and test

train_data= df1.copy()

test_data = df2.copy()
#Work on the collinearity and drop the features as per Observation 3

drop_cols = ['MSSubClass', 'Exterior1st', 'GarageYrBlt','1stFlrSF', '2ndFlrSF','TotRmsAbvGrd','GarageArea']

for i in drop_cols:

    train_data.drop(i, axis=1, inplace=True)

    test_data.drop(i, axis=1, inplace=True)
#All set done, let us now check the basic Linear Regression and OLS model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import statsmodels.api as sms
X = train_data.iloc[:,:-1]

y = train_data.iloc[:,-1]

print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
linReg = LinearRegression()

linReg.fit(X_train,y_train)

y_lin_pred=linReg.predict(X_test)

print('Mean Squared Error is ',metrics.mean_squared_error(y_test,y_lin_pred))

print('R2 value is ',metrics.r2_score(y_test,y_lin_pred))
sm_mod = sms.OLS(y,X).fit()

print(sm_mod.summary())
P_val_cols1 = ['MSZoning','LandContour','Utilities','LotConfig','Neighborhood',

'Condition1','BldgType','RoofStyle','RoofMatl','Exterior2nd',

'MasVnrType','MasVnrArea','ExterQual','Foundation','BsmtCond','BsmtFinSF1','BsmtUnfSF',

'Heating','Electrical','LowQualFinSF','BsmtHalfBath','KitchenAbvGr','GarageFinish',

'GarageQual','GarageCond','OpenPorchSF','3SsnPorch','PoolQC','Fence','MiscVal','MoSold','SaleType']



for i in P_val_cols1:

    train_data.drop(i, axis=1, inplace=True)

    test_data.drop(i, axis=1, inplace=True)
X1= train_data.iloc[:,:-1]

y1=train_data.iloc[:,-1]

X1_train, X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=42)
lin_mod =linReg.fit(X1_train,y1_train)

y1_lin_pred=lin_mod.predict(X1_test)

print('Mean Squared Error is ',metrics.mean_squared_error(y1_test,y1_lin_pred))

print('R2 value is ',metrics.r2_score(y1_test,y1_lin_pred))
from sklearn.linear_model import Ridge, Lasso

rr = Ridge(alpha=0.01)

rr_mod=rr.fit(X1_train,y1_train)

y2_ridg_pred=rr_mod.predict(X1_test)

print('Mean Squared Error is ',metrics.mean_squared_error(y1_test,y2_ridg_pred))

print('R2 value is ',metrics.r2_score(y1_test,y2_ridg_pred))
lso = Lasso()

lso_mod=lso.fit(X1_train,y1_train)

y3_lso_pred=lso_mod.predict(X1_test)

print('Mean Squared Error is ',metrics.mean_squared_error(y1_test,y3_lso_pred))

print('R2 value is ',metrics.r2_score(y1_test,y3_lso_pred))
from sklearn.ensemble import GradientBoostingRegressor

grb = GradientBoostingRegressor()

grb_mod= grb.fit(X1_train, y1_train)

y4_grb_pred=grb_mod.predict(X1_test)

print('Mean Squared Error is ',metrics.mean_squared_error(y1_test,y4_grb_pred))

print('R2 value is ',metrics.r2_score(y1_test,y4_grb_pred))
#Predicting the test value now into the test data set

test_grb_pred = grb_mod.predict(test_data)

print(test_grb_pred)
# Ahh.. the data was log transformed, we need to convert them back to normal

result = pd.DataFrame({'Id':test_data['Id'],'SalePrice':np.exp(test_grb_pred)})

result.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")