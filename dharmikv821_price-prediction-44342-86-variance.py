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

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

%matplotlib inline

pd.set_option('display.max_columns',None)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
train.shape
train.SalePrice.describe()
train.info()
na_cols = train.columns[train.isna().any()]

for col in na_cols:

    print(col,":",np.round(train[col].isna().mean()*100,3),'\tCount :', train[col].isna().sum())
for col in na_cols:

    data = train.copy()

    data[col] = np.where(data[col].isna(),1,0)

    print(data.groupby(col)['SalePrice'].median())

    print(np.round(train[col].isna().mean()*100,3),'%')

    data.groupby(col)['SalePrice'].median().plot.bar()

    plt.title(col)

    plt.show()
num_cols = [col for col in train.columns if train[col].dtype !='O']

print(len(num_cols))

train[num_cols].head()
for col in num_cols:

    sns.boxplot(train[col])

    plt.title(col)

    plt.xlabel(col)

    plt.show()  
year_col = [col for col in num_cols if 'Year' in col or 'Yr' in col]

year_col
print("UNIQUE Value Count : \n",train[year_col].nunique())

print('\nNULL Value Count : \n',train[year_col].isna().sum())

print('\nUNIQUE Values :\n')

for col in year_col:

    print(col,':',train[col].unique())
train.groupby('YrSold')['SalePrice'].median().plot()

plt.ylabel('SalePrice')

plt.title('YrSold Vs. SalePrice')

plt.show()
for col in year_col:

    if col != 'YrSold':

        data = train.copy()

        data[col] = data['YrSold']-data[col]

        plt.scatter(data[col],data['SalePrice'],)

        plt.title(col.upper())

        plt.xlabel(col)

        plt.ylabel('SalePrice')

        plt.show()

        
disc_cols = [col for col in num_cols if len(train[col].unique())<25 and col not in year_col]

disc_cols
train[disc_cols].head()
for col in disc_cols:

    data = train.copy()

    data.groupby(col)['SalePrice'].median().plot.bar()

    plt.xlabel(col)

    plt.ylabel('SalePrice')

    plt.title(col.upper())

    plt.show()
cont_cols = [col for col in num_cols if col not in disc_cols + year_col + ['Id']]

for col in cont_cols:

    data = train.copy()

    data[col].hist(bins=25)

    plt.xlabel(col)

    plt.ylabel('Count')

    plt.title(col.upper())

    plt.show()
for col in cont_cols:

    data = train.copy()

    if 0 in data[col].unique():

        pass

    else:

        data[col] = np.log(data[col])

        data['SalePrice'] = np.log(data['SalePrice'])

        plt.scatter(data[col],data['SalePrice'])

        plt.xlabel(col)

        plt.ylabel('SalePrice')

        plt.title(col.upper())

        plt.show()
cat_cols = [col for col in train.columns if train[col].dtypes=='O']

cat_cols
train[cat_cols].head()
train[cat_cols].nunique()
for col in cat_cols:

    data = train.copy()

    data.groupby(col)['SalePrice'].median().plot.bar()

    plt.xlabel(col)

    plt.ylabel('SalePrice')

    plt.title(col)

    plt.show()
na_cat_cols = [col for col in cat_cols if train[col].isna().sum()>1]

for col in na_cat_cols:

    print(col,':',np.round(train[col].isna().mean()*100,3),'%')
def rep_cat_cols(data, na_col):

    df = data.copy()

    df[na_col] = df[na_col].fillna('Missing')

    return df
train = rep_cat_cols(train, na_cat_cols)

train[na_cat_cols].isna().sum()
na_num_cols = [col for col in num_cols if train[col].isna().sum()>1]

for col in na_num_cols:

    print(col,':',np.round(train[col].isna().mean()*100,3),'%')
for col in na_num_cols:

    med_val = train[col].median()

    train[col+'_na'] = np.where(train[col].isna(),1,0)

    train[col].fillna(med_val, inplace=True)

train[na_num_cols].isna().sum()
train.head()
for col in year_col:

    if col!='YrSold':

        train[col] = train['YrSold']-train[col]
train[year_col].head()
for col in ['LotFrontage','LotArea', '1stFlrSF','GrLivArea','SalePrice']:

    train[col]=np.log(train[col])
train.head()
for col in cat_cols:

    temp = train.groupby(col)['SalePrice'].count()/len(train)

    temp_df = temp[temp>0.01].index

    train[col] = np.where(train[col].isin(temp_df),train[col],'Rare_var')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:

    train[col] = le.fit_transform(train[col])

train.head()
features = [col for col in train.columns if col not in ['Id','SalePrice']]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train[features] = sc.fit_transform(train[features])
train.head(10)
X = train.drop(['Id','SalePrice'],axis=1)

y = train['SalePrice']
X
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(Lasso(alpha=0.005,random_state=0))

selector.fit(X,y)

len(X.columns[selector.get_support()])
selector.get_support()
x = X[X.columns[selector.get_support()]]
x.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(X_train,y_train)

pred = xgb.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(np.exp(y_test), np.exp(pred)))

print(mean_squared_error(np.exp(y_test), np.exp(pred)))

print(np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred))))
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test.head()
na_cols = test.columns[test.isna().any()]

len(na_cols)
num_cols = [col for col in test.columns if test[col].dtype !='O']

len(num_cols)
na_cat_cols = [col for col in cat_cols if test[col].isna().sum()>1]

test = rep_cat_cols(test, na_cat_cols)

test[na_cat_cols].isna().sum()

na_num_cols = [col for col in num_cols if test[col].isna().sum()>1]

for col in na_num_cols:

    med_val = test[col].median()

    test[col+'_na'] = np.where(test[col].isna(),1,0)

    test[col].fillna(med_val, inplace=True)

for col in year_col:

    if col!='YrSold':

        test[col] = test['YrSold']-test[col]

for col in ['LotFrontage','LotArea', '1stFlrSF','GrLivArea']:

    test[col]=np.log(test[col])

for col in cat_cols:

    temp = train.groupby(col)['SalePrice'].count()/len(train)

    temp_df = temp[temp>0.01].index

    test[col] = np.where(test[col].isin(temp_df),test[col],'Rare_var')

for col in cat_cols:

    test[col] = le.transform(test[col])

test[features] = sc.transform(test[features])

x_test = test.drop(['Id'],axis=1)

x_test = x_test[X.columns[selector.get_support()]]
x_test.columns[x_test.isna().any()]
x_test.isna().sum()
x_test['GarageCars'].fillna(x_test['GarageCars'].mode()[0],inplace=True)
x_test['BsmtFinSF1'].fillna(x_test['BsmtFinSF1'].median(),inplace=True)
x_test['TotalBsmtSF'].fillna(x_test['TotalBsmtSF'].median(),inplace=True)
print(x_test.isna().sum())
submit = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submit.head()
prediction = xgb.predict(x_test)

prediction = np.exp(prediction)

print(mean_absolute_error(submit.SalePrice, prediction))

print(mean_squared_error(submit.SalePrice, prediction))

print(np.sqrt(mean_squared_error(submit.SalePrice, prediction)))