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
train_df = pd.read_csv('../input/train_2016_v2.csv')
# No null entries - nothing to fill

train_df.info()
train_df.columns
train_df.head()
## there are 352 unique dates

train_df['transactiondate'].nunique()
train_df['logerror'].describe()
import seaborn as sns
sns.boxplot(train_df['logerror'])
train_df[train_df['logerror'] < -1].count()
import re
train_df['year'] = train_df['transactiondate'].apply(lambda X: X.split('-')[0])
train_df['month'] = train_df['transactiondate'].apply(lambda X: X.split('-')[1])

train_df['day'] = train_df['transactiondate'].apply(lambda X: X.split('-')[2])
train_df.head()
del train_df['transactiondate']
train_df.groupby(['year', 'month'])['parcelid'].aggregate('count')
prop_df = pd.read_csv('../input/properties_2016.csv')
prop_df.head()
prop_df.columns
prop_df.info()
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()
train_df.isnull().sum()
high_null_cols_df = pd.DataFrame((train_df.isnull().sum() > (train_df.shape[0] / 2)).reset_index())
high_null_cols_df.columns = ['category', 'high_null']
high_null_cols = []

for i in range (0, high_null_cols_df.shape[0]):

    if (high_null_cols_df.iloc[i]['high_null'] == True):

        high_null_cols.append(high_null_cols_df.iloc[i]['category'])

    else:

        pass
high_null_cols
train_df.drop(high_null_cols, axis=1, inplace=True)

##drop all columns with too many nulls
train_df['error'] = np.exp(train_df['logerror'])
train_df.isnull().sum()
train_df.groupby('bedroomcnt')['error'].mean()
train_df.groupby('bathroomcnt')['error'].mean()
train_df['latitude_scaled'] = np.rint(train_df['latitude']/10000)
train_df['longitude_scaled'] = np.rint(train_df['longitude']/100000)
train_df['year'] = train_df['year'].apply(lambda X: float(X))

train_df['month'] = train_df['month'].apply(lambda X: float(X))

train_df['day'] = train_df['day'].apply(lambda X: float(X))
train_df['propertycountylandusecode'].value_counts()
train_df['propertycountylandusecode'].fillna(value='0100', inplace=True)
prop_land_codes = train_df['propertycountylandusecode'].unique()

prop_land_codes_zf = dict(zip(prop_land_codes, range(len(prop_land_codes))))
train_df['propertycountylandusecode'].replace(prop_land_codes_zf, inplace=True)
prop_zone_desc = train_df['propertyzoningdesc']
del train_df['propertyzoningdesc']
train_df.isnull().sum()
train_df['buildingqualitytypeid'].value_counts()
del train_df['calculatedbathnbr']
del train_df['finishedsquarefeet12']
train_df['yearbuilt'].value_counts().head()
train_df[train_df['bathroomcnt'] == train_df['fullbathcnt'] + 0]['error'].mean()
train_df[train_df['bathroomcnt'] == train_df['fullbathcnt'] + 0.5]['error'].mean()
train_df[train_df['bathroomcnt'] == train_df['fullbathcnt'] + 1]['error'].mean()
(train_df['bathroomcnt'] - train_df['fullbathcnt']).value_counts()
train_df['fullbathcnt'].fillna(train_df['bathroomcnt'], inplace=True)
del train_df['censustractandblock']
tax_df = pd.DataFrame(train_df.groupby('roomcnt')['taxamount'].aggregate('mean').reset_index())
tax_df
train_df = pd.merge(train_df, tax_df, on='roomcnt', how='left')
train_df['taxamount_x'].fillna(train_df['taxamount_y'], inplace=True)
del train_df['taxamount_y']
train_df.isnull().sum()
del train_df['buildingqualitytypeid']
train_df['heatingorsystemtypeid'].fillna(method='ffill', inplace=True)
train_df['lot_to_house_ratio'] = train_df['lotsizesquarefeet']/train_df['calculatedfinishedsquarefeet']
import matplotlib.pyplot as plt
train_df['lot_to_house_ratio'].describe()
train_df['lotsizesquarefeet'].fillna(4.578765 * train_df['calculatedfinishedsquarefeet'], inplace=True)
train_df.isnull().sum()
train_df.groupby('unitcnt')['calculatedfinishedsquarefeet'].aggregate('mean')
def return_unitcnt(X):

    if(X > 2493):

      return 3.0

    elif(X > 1976.0):

      return 2.0

    else:

       return 1.0
train_df['temp'] = train_df['calculatedfinishedsquarefeet'].apply(lambda X:return_unitcnt(X))
train_df['unitcnt'].fillna(train_df['temp'], inplace=True)
train_df.isnull().sum()
del train_df['lot_to_house_ratio']
del train_df['temp']
train_df['regionidcity'].value_counts().head(5)
train_df['regionidcity'].fillna(12447.0, inplace=True)
train_df.groupby('bedroomcnt')['calculatedfinishedsquarefeet'].aggregate('mean')
train_df[train_df['calculatedfinishedsquarefeet'].isnull()]['bedroomcnt'].value_counts()
def fill_sqft(X):

    if(X == 0):

        return 1914

    elif(X == 1):

        return 819

    elif(X == 2):

        return 1209

    else:

        return 1633
train_df['temp'] = train_df['bedroomcnt'].apply(lambda X:fill_sqft(X))
train_df['calculatedfinishedsquarefeet'].fillna(train_df['temp'], inplace=True)
del train_df['temp']
train_df.isnull().sum()
train_df.groupby('bedroomcnt')['lotsizesquarefeet'].aggregate('mean').head(5)
train_df[train_df['lotsizesquarefeet'].isnull()]['bedroomcnt'].value_counts()
def lotSqFt(X):

    if(X == 0):

        return 55528

    else:

        return 21658
train_df['temp'] = train_df['bedroomcnt'].apply(lambda X:lotSqFt(X))
train_df['lotsizesquarefeet'].fillna(train_df['temp'], inplace=True)
train_df['yearbuilt'].fillna(method='bfill', inplace=True)
train_df[train_df['regionidzip'].isnull()]['regionidcounty'].value_counts()
train_df[train_df['regionidcounty'] == 2061.0]['regionidzip'].value_counts().head(5)
train_df[train_df['regionidcounty'] == 1286.0]['regionidzip'].value_counts().head(5)
def fill_zip(X):

    if(X == 2061.0):

        return 97118.0

    else:

        return 96987.0
del train_df['temp']
train_df['temp'] = train_df['regionidcounty'].apply(lambda X:fill_zip(X))
train_df['regionidzip'].fillna(train_df['temp'], inplace=True)
import seaborn as sns
sns.lmplot(data=train_df, x='taxvaluedollarcnt', y='structuretaxvaluedollarcnt')
del train_df['structuretaxvaluedollarcnt']
tax_val_mean = train_df['taxvaluedollarcnt'].mean()
train_df['taxvaluedollarcnt'].fillna(train_df['taxvaluedollarcnt'].mean(), inplace=True)
train_df['landtaxvaluedollarcnt'].fillna(train_df['landtaxvaluedollarcnt'].mean(), inplace=True)
train_df['age'] = 2016 - train_df['yearbuilt']
del train_df['yearbuilt']
from sklearn.model_selection import train_test_split
X = train_df.drop(['logerror', 'error', 'temp'], axis=1)
Y = train_df['error']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)
print(lm.coef_)
error_pred = lm.predict(X_test)
plt.scatter(Y_test, error_pred)

plt.xlabel('Y_test')

plt.ylabel('Pred_y')

plt.show()
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(Y_test, error_pred))

print('MSE:', metrics.mean_squared_error(Y_test, error_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, error_pred)))
plt.scatter(np.log(Y_test), np.log(error_pred))

plt.xlabel('Y_test')

plt.ylabel('Pred_y')

plt.show()