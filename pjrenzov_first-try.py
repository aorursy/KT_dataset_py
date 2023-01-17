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
import pandas as pd

import numpy as np

import seaborn as sns
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
target = train['SalePrice']

train = train.drop({'Id'}, axis = 1)
test = test.drop({'Id'}, axis = 1)
print(train.shape)

print(test.shape)
import matplotlib.pyplot as plt
fig,((ax1,ax2),(ax3,ax4),(ax5,ax6))= plt.subplots(3,2,sharex=False,sharey=True)

ax1.scatter(train['MSSubClass'],target)

ax1.set_title('MSSubClass')

ax2.scatter(train['MSZoning'],target)

ax2.set_title('MSZoning')

ax3.scatter(train["LotArea"],target)

ax3.set_title('LotArea')

ax4.scatter(train['Street'],target)

ax4.set_title('Street')

ax5.scatter(train['LotShape'],target)

ax5.set_title('LotShape')

ax6.scatter(train['LandContour'],target)

ax6.set_title('LandContour')
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(total, percent)
train = train.drop((missing_data[missing_data['Total'] > 1]).index,axis =1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)
train = train.drop([1298,523])
target = train['SalePrice']
train.head()
corrmat = train.corr()

sns.heatmap(corrmat)
corrmat
cols = []

for i in range(34):

    if(corrmat.iloc[i, 33] >= 0.2):

        cols.append(corrmat.index[i])
for col in cols:

    plt.figure()

    sns.scatterplot(x = train[col] , y = train['SalePrice'])
df = train[cols]
corrmat_1 = df.corr()

sns.heatmap(corrmat_1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
cols_2 = []

for key in train.columns:

    if (np.dtype(train[key]) == object):

        cols_2.append(key)

df_2 = train[cols_2]
for col in cols_2:

    df_2[col] = labelencoder.fit_transform(df_2[col])

df_2 
df_2['SalePrice'] = train['SalePrice']
corrmat_3 = df_2.corr()

corrmat_3
cols_3 = []

for i in range(29):

    if(corrmat_3.iloc[i, 28] >= 0.2):

        cols_3.append(corrmat_3.index[i])
cols_3
train['Electrical'].unique()
test['Electrical'].unique()
train_f = pd.DataFrame()

for col in cols:

    train_f[col] = train[col]

for col in cols_3:

    train_f[col] = df_2[col]
train_f
target = train_f['SalePrice']

train_f = train_f.drop({'SalePrice'}, axis = 1)
train_f.info()
train_col = []

for col in cols_3 :

    if col!='SalePrice':

        train_col.append(np.max(train_f[col]))
from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder(categorical_features = [19,20,21,22,23,24,25])

train_f = oh.fit_transform(train_f).toarray()
train_f = pd.DataFrame(train_f)
train_f.head()
train_f.iloc[0,53]
test_f = pd.DataFrame()

for col in cols:

    if col != 'SalePrice':

        test_f[col] = test[col]

for col in cols_3:

    if col != 'SalePrice':

        test_f[col] = test[col]
np.zeros((1,26))
test_f.info()
for col in cols_3:

    if col != 'SalePrice':

        test_f[col] = labelencoder.fit_transform(test_f[col])
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values = 'NaN', strategy = 'mean' )

test_f = imp.fit_transform(test_f)
test_f = pd.DataFrame(test_f)

test_f.info()
#test_f.loc[1459] = pd.DataFrame(np.zeros((1,26)))

#test_f.iloc[1459,-3] = int(4)

#test_f.fillna(0)

test_f.loc[1459] = np.zeros(26)

for i in range(26):

    test_f.iloc[1459,i] = 0

test_f.iloc[1459,-3] = int(4)
test_col = []

for i in range(19,26) :

    test_col.append(np.max(test_f.iloc[:, i]))
test_col
train_col
oh_1 = OneHotEncoder(categorical_features = [-1,-2,-3,-4,-5,-6,-7])

test_f = oh_1.fit_transform(test_f).toarray()
train_f.shape
test_f.shape
test_f = np.delete(test_f, 1459, 0)
test_f.shape
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(train_f, target)

predict = regressor.predict(test_f)

predict
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission['SalePrice'] = predict

submission.to_csv('submission.csv', index = False)