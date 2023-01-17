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
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_train.head()
df_train.describe().transpose()
df_train.columns
df_train['SalePrice'].describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(10,6))

sns.distplot(df_train['SalePrice'])
df_train.corr()['SalePrice'].sort_values()
plt.figure(figsize=(10,6))

sns.countplot(x='OverallQual',data=df_train,palette='viridis')
plt.figure(figsize=(8,6))

sns.scatterplot(x='SalePrice',y='GrLivArea',data=df_train) 

plt.tight_layout()
plt.figure(figsize=(8,6))

sns.scatterplot(x='SalePrice',y='TotalBsmtSF',data=df_train) 

plt.tight_layout()
plt.figure(figsize=(12,5))

sns.boxplot(x='BedroomAbvGr',y='SalePrice',data=df_train)
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
# we can conclude that:



# 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. 

# Both relationships are positive, which means that as one variable increases, the other also increases.

# In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.

# 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'.

# The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the 

# overall quality.



# We just analysed four variables, but there are many other that we should analyse.

# The trick here seems to be the choice of the right features (feature selection) 

# and not the definition of complex relationships between them (feature engineering).
plt.figure(figsize=(12,9))

sns.heatmap(df_train.corr() ,vmax=.8,square=True)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# Let's analyse this to understand how to handle the missing data.



# We'll consider that when more than 15% of the data is missing, we should delete the corresponding variable and pretend it never existed. This means that we will not try any trick to fill the missing data in these cases. According to this, there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.) that we should delete. The point is: will we miss this data? I don't think so. None of these variables seem to be very important, since most of them are not aspects in which we think about when buying a house (maybe that's the reason why data is missing?). Moreover, looking closer at the variables, we could say that variables like 'PoolQC', 'MiscFeature' and 'FireplaceQu' are strong candidates for outliers, so we'll be happy to delete them.



# In what concerns the remaining cases, we can see that 'GarageX' variables have the same number of missing data. I bet missing data refers to the same set of observations (although I will not check it; it's just 5% and we should not spend 20in5



# problems). Since the most important information regarding garages is expressed by 'GarageCars' and considering that we are just talking about 5% of missing data, I'll delete the mentioned 'GarageX' variables. The same logic applies to 'BsmtX' variables.



# Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. Furthermore, they have a strong correlation with 'YearBuilt' and 'OverallQual' which are already considered. Thus, we will not lose information if we delete 'MasVnrArea' and 'MasVnrType'.



# Finally, we have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation and keep the variable.



# In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
import pandas_profiling as pp
pp.ProfileReport(df_train)
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
y = df_train[['Id','SalePrice']]

df_train = df_train.drop('SalePrice',axis=1)
all_dfs = [df_train,df_test]

all_df = pd.concat(all_dfs).reset_index(drop=True);
display_all(all_df.isnull().sum()/all_df.shape[0])
all_df.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],axis=1,inplace=True)
all_df['LotFrontage'].fillna(value=all_df['LotFrontage'].median(),inplace=True)

all_df['MasVnrType'].fillna(value='None',inplace=True)

all_df['MasVnrArea'].fillna(0,inplace=True)

all_df['BsmtCond'].fillna(value='TA',inplace=True)

all_df['BsmtExposure'].fillna(value='No',inplace=True)

all_df['Electrical'].fillna(value='SBrkr',inplace=True)

all_df['BsmtFinType2'].fillna(value='Unf',inplace=True)

all_df['GarageType'].fillna(value='Attchd',inplace=True)

all_df['GarageYrBlt'].fillna(value=all_df['GarageYrBlt'].median(),inplace=True)

all_df['GarageFinish'].fillna(value='Unf',inplace=True)

all_df['GarageQual'].fillna(value='TA',inplace=True)

all_df['GarageCond'].fillna(value='TA',inplace=True)

all_df['BsmtFinType1'].fillna(value='NO',inplace=True)

all_df['BsmtQual'].fillna(value='No',inplace=True)

all_df['BsmtFullBath'].fillna(value=all_df['BsmtFullBath'].median(),inplace=True)

all_df['BsmtFinSF1'].fillna(value=all_df['BsmtFinSF1'].median(),inplace=True)

all_df['BsmtFinSF2'].fillna(value=0,inplace=True)

all_df['BsmtUnfSF'].fillna(value=0,inplace=True)

all_df['TotalBsmtSF'].fillna(value=all_df['TotalBsmtSF'].median(),inplace=True)

all_df['BsmtHalfBath'].fillna(value=0,inplace=True)

all_df['GarageCars'].fillna(value=all_df['GarageCars'].median(),inplace=True)

all_df['GarageArea'].fillna(value=all_df['GarageArea'].median(),inplace=True)
from sklearn.preprocessing import StandardScaler, LabelEncoder
labelencoder=LabelEncoder()



all_df['MSZoning']      = labelencoder.fit_transform(all_df['MSZoning'].astype(str))

all_df['Exterior1st']   = labelencoder.fit_transform(all_df['Exterior1st'].astype(str))

all_df['Exterior2nd']   = labelencoder.fit_transform(all_df['Exterior2nd'].astype(str))

all_df['KitchenQual']   = labelencoder.fit_transform(all_df['KitchenQual'].astype(str))

all_df['Functional']    = labelencoder.fit_transform(all_df['Functional'].astype(str))

all_df['SaleType']      = labelencoder.fit_transform(all_df['SaleType'].astype(str))

all_df['Street']        = labelencoder.fit_transform(all_df['Street'])   

all_df['LotShape']      = labelencoder.fit_transform(all_df['LotShape'])   

all_df['LandContour']   = labelencoder.fit_transform(all_df['LandContour'])   

all_df['LotConfig']     = labelencoder.fit_transform(all_df['LotConfig'])   

all_df['LandSlope']     = labelencoder.fit_transform(all_df['LandSlope'])   

all_df['Neighborhood']  = labelencoder.fit_transform(all_df['Neighborhood'])   

all_df['Condition1']    = labelencoder.fit_transform(all_df['Condition1'])   

all_df['Condition2']    = labelencoder.fit_transform(all_df['Condition2'])   

all_df['BldgType']      = labelencoder.fit_transform(all_df['BldgType'])   

all_df['HouseStyle']    = labelencoder.fit_transform(all_df['HouseStyle'])   

all_df['RoofStyle']     = labelencoder.fit_transform(all_df['RoofStyle'])   

all_df['RoofMatl']      = labelencoder.fit_transform(all_df['RoofMatl'])    

all_df['MasVnrType']    = labelencoder.fit_transform(all_df['MasVnrType'])   

all_df['ExterQual']     = labelencoder.fit_transform(all_df['ExterQual'])  

all_df['ExterCond']     = labelencoder.fit_transform(all_df['ExterCond'])   

all_df['Foundation']    = labelencoder.fit_transform(all_df['Foundation'])   

all_df['BsmtQual']      = labelencoder.fit_transform(all_df['BsmtQual'])   

all_df['BsmtCond']      = labelencoder.fit_transform(all_df['BsmtCond'])   

all_df['BsmtExposure']  = labelencoder.fit_transform(all_df['BsmtExposure'])   

all_df['BsmtFinType1']  = labelencoder.fit_transform(all_df['BsmtFinType1'])   

all_df['BsmtFinType2']  = labelencoder.fit_transform(all_df['BsmtFinType2'])   

all_df['Heating']       = labelencoder.fit_transform(all_df['Heating'])   

all_df['HeatingQC']     = labelencoder.fit_transform(all_df['HeatingQC'])   

all_df['CentralAir']    = labelencoder.fit_transform(all_df['CentralAir'])   

all_df['Electrical']    = labelencoder.fit_transform(all_df['Electrical'])    

all_df['GarageType']    = labelencoder.fit_transform(all_df['GarageType'])  

all_df['GarageFinish']  = labelencoder.fit_transform(all_df['GarageFinish'])   

all_df['GarageQual']    = labelencoder.fit_transform(all_df['GarageQual'])  

all_df['GarageCond']    = labelencoder.fit_transform(all_df['GarageCond'])   

all_df['PavedDrive']    = labelencoder.fit_transform(all_df['PavedDrive'])  

all_df['SaleCondition'] = labelencoder.fit_transform(all_df['SaleCondition'])
Scaler = StandardScaler()

all_scaled = pd.DataFrame(Scaler.fit_transform(all_df))



train_scaled = pd.DataFrame(all_scaled[:1460])

test_scaled = pd.DataFrame(all_scaled[1460:2920])
from sklearn.model_selection import train_test_split

X = train_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y['SalePrice'], test_size=0.1, random_state=101)
from xgboost import XGBRegressor

XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)

XGB.fit(X_train,y_train)
from lightgbm import LGBMRegressor

LGBM = LGBMRegressor(n_estimators = 1000)

LGBM.fit(X_train,y_train)
print ("Training score(XGB):",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))

print ("Training score(LGBM):",LGBM.score(X_train,y_train),"Test Score:",LGBM.score(X_test,y_test))

y_pred_xgb  = pd.DataFrame( XGB.predict(test_scaled))

y_pred_lgbm = pd.DataFrame(LGBM.predict(test_scaled))



y_pred=pd.DataFrame()

y_pred['SalePrice'] = 0.5 * y_pred_xgb[0] + 0.5 * y_pred_lgbm[0]

y_pred['Id'] = df_test['Id']
y_pred.to_csv('house_price_blend.csv',index=False)