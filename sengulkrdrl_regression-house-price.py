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
import seaborn as sns

import matplotlib as plt

import matplotlib.pyplot as plt

import matplotlib.style as style

from scipy import stats 

sample = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test =  pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.info()
train.columns
train.describe()
train["SalePrice"].describe()
style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))



mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

plt.title("Heatmap of all the Features", fontsize = 30);
def string_remover(df,list1=[],drop=[]):

    a = df.select_dtypes(include='object')

    for i in a.columns:

        for x in a.index:

            try:

                c = list1.index(a[i].iloc[x:x+1][x])

                a[i].iloc[x:x+1][x] = c

            except:

                list1.append(a[i].iloc[x:x+1][x])

                a[i].iloc[x:x+1][x] = len(list1)-1

    a.fillna(len(list1))

    d = df.select_dtypes(exclude='object').fillna(0)

    try:

        return pd.concat([d,a],axis=1).fillna(0).drop([drop],axis=1)

    except:

        return pd.concat([d,a],axis=1).fillna(0)
b = []

train = string_remover(train,list1=b)

test = string_remover(test,list1=b)
train.head()
test.head()
import pandas_profiling as pp

pp.ProfileReport(train)
sns.set_style('darkgrid')



fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.distplot(train['SalePrice'], ax=ax)



ax.set_xlabel('Price(USD)')

plt.suptitle('Distribution of Price', size=10)

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train.drop('SalePrice',axis=1),train['SalePrice'],test_size=0.33,random_state=42)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train,y_train)
forest_pred = forest.predict(test)
sample['SalePrice'] = pd.Series(forest_pred)

sample.to_csv('submission_forest.csv',index=False)

sample
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
reg_pred = lm.predict(X_test)
plt.scatter(y_test, reg_pred, color ="blue")
sns.distplot((y_test-reg_pred),bins=50)
from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, reg_pred))

print('MSE:', metrics.mean_squared_error(y_test, reg_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, reg_pred)))
sample['SalePrice'] = pd.Series(reg_pred)

sample.to_csv('submission.csv',index=False)

sample