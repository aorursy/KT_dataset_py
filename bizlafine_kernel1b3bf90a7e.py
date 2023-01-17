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
import seaborn as sns

# Plotting.
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (10, 6)
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
print(len(df))
df.head(3)
df.sample(5)
len(df.columns)
df.SalePrice.plot.hist(bins=50)
matr_corr = df.corr()
price_corr = matr_corr['SalePrice'].sort_values(ascending=False)
top_features = price_corr[price_corr.abs() > 0.4].index
print(top_features)

sns.heatmap(matr_corr.loc[top_features, top_features])

sns.violinplot(x='OverallQual', y='SalePrice', data=df)
df.GrLivArea.plot.hist(bins=50)
df.plot.scatter(x='GrLivArea', y='SalePrice')
df.SalePrice.plot.hist(bins=50)
df.SalePrice.apply(np.log).plot.hist(bins=50)
def preprocess(raw_df) :
    df = raw_df.copy()
    if 'SalePrice' in df.columns :
        df['SalePricelog'] = np.log(df.SalePrice)
    cat_variables = df.select_dtypes(include=['object']).columns
    cat_df = pd.get_dummies(df[cat_variables])
    num_df = df[[c for c in df if c not in cat_variables]].fillna(0.0)
    return pd.concat([cat_df, num_df], axis=1)
train_set = preprocess(df)
train_set.head(3)
from sklearn.preprocessing import normalize
features_df = train_set.copy().fillna(0.0)

target = features_df.pop('SalePricelog')
del features_df['SalePrice']

features_df.sample(3)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features_df, target, test_size=0.25, random_state=0)
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

model = RidgeCV()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(len(y_pred))
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, y_pred))
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_df.head(3)
test_set = preprocess(test_df)
test_set.head(3)
for c in features_df.columns:
  if c not in test_set.columns:
    test_set[c] = 0

for c in test_set.columns:
  if c not in features_df.columns:
    del test_set[c]

test_df = test_set[features_df.columns]
predictions = np.exp(model.predict(test_df))
sub_df = pd.DataFrame(dict(Id=test_df.Id, SalePrice=predictions))
sub_df
sub_df.to_csv("‪submission.csv", index=False)
