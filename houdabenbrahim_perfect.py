# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #still don't know what it does reallyyyy 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#plotting :

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (10, 6)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
print(df.shape)
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df.sample(10)
df.head(5)
df.columns
len(df)
#problem of skewed 

df.SalePrice.plot.hist(bins=100)
#les valeurs correlés 



corr_mat = df.corr()

price_corr = corr_mat['SalePrice']

top_corr_features = price_corr[price_corr.abs() > 0.4].index



sns.heatmap(corr_mat.loc[top_corr_features, top_corr_features])
price_corr[price_corr.abs() > 0.4].sort_values(ascending=False)
sns.violinplot(x='OverallQual', y='SalePrice', data=df)
df.GrLivArea.plot.hist(bins=100)
df.plot.scatter(x='GrLivArea', y='SalePrice')
#log pour eliminer les valeurs extremes 

df.SalePrice.plot.hist(bins=100)
df.SalePrice.apply(np.log).plot.hist(bins=100)
#on va créer notre modèle : RQQQQ:on dopit enlever saleprice car c'est pour lui qu'on va faire la prediction

#on l'a laisser just epour faire le training 

def preprocess(raw_df):

  df = raw_df.copy()

  if 'SalePriceLog'   in df.columns:

    df['SalePriceLog'] = np.log(df.SalePrice)

  cat_variables = df.select_dtypes(include=['object']).columns

  cat_df = pd.get_dummies(df[cat_variables])

  num_df = df[[c for c in df if c not in cat_variables]].fillna(0.0)

  #num_df = (num_df - num_df.mean()) / num_df.std()

  return pd.concat([cat_df, num_df], axis=1) 
train_df = preprocess(df)

train_df.sample(3)
#on les a enlever tous les deux saleprice et salepricelog

from sklearn.preprocessing import normalize

features_df = train_df.copy().fillna(0.0)

#features_df = train_df[['OverallQual', 'GrLivArea', 'SalePriceLog']].copy()

#features_df['GrLivArea'] = (features_df.GrLivArea - features_df.GrLivArea.mean()) / features_df.GrLivArea.std()



#features_df = (features_df - features_df.mean()) / features_df.std()



target = features_df.pop('SalePriceLog')

del features_df['SalePrice']

features_df.sample(3)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(features_df, target, test_size=0.25, random_state=0)
from sklearn.metrics import mean_squared_error

from sklearn.dummy import DummyRegressor

baseline = DummyRegressor('mean')

baseline.fit(x_train, y_train)

np.sqrt(mean_squared_error(y_test, baseline.predict(x_test)))
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

model = RidgeCV()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test, y_pred))
plt.scatter(y_test, y_pred, c='g')

plt.scatter(y_train, model.predict(x_train), c='r', alpha=0.2)

plt.xlabel("Sale price")

plt.ylabel("Predicted price")

#plt.plot([10., 14.], [10., 14.], '--')
features_df = train_df.copy()

target = features_df.pop('SalePriceLog')

del features_df['SalePrice']



model = RidgeCV()

model.fit(features_df, target)
raw_test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df = preprocess(raw_test_df)
for c in features_df.columns:

  if c not in test_df.columns:

    test_df[c] = 0



for c in test_df.columns:

  if c not in features_df.columns:

    del test_df[c]



test_df = test_df[features_df.columns]
predicted_price = np.exp(model.predict(test_df))
predicted_price
sub_df = pd.DataFrame(dict(Id=test_df.Id, SalePrice=predicted_price))
sub_df
print(predicted_price)
sub_df = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predicted_price})



sub_df.to_csv('submission.csv', index=False)