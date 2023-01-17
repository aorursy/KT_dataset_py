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
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")

print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
train.info()
test.info()
train = train.fillna(0)

test = test.fillna(0)
print(train.isna().sum())

print(test.isna().sum())
def objtypelist(df):

    objecttype=[]

    for col in df.columns:

        if(df[col].dtype == np.float64 or df[col].dtype == np.int64):

            pass

        else:

            objecttype.append(col)

    return objecttype
train_obj = objtypelist(train)
test_obj = objtypelist(test)
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train.corr()['SalePrice'].sort_values()
plt.figure(figsize=(20,10))

sns.heatmap(train.corr(), annot=True, cmap="Reds")

plt.show()
X = train[['OverallQual']]

y = train['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)
from sklearn.linear_model import LinearRegression
mod = LinearRegression()
mod.fit(X_train, y_train)
mod.intercept_, mod.coef_
from sklearn.metrics import r2_score,mean_squared_error
y_train_pred = mod.predict(X_train)

r2_score(y_train, y_train_pred)
test['SalePrice'] = -90334.15280801835 + 44467.70101543 * test['OverallQual']

test['SalePrice']
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test['SalePrice']})

my_submission.to_csv('submission.csv', index=False)