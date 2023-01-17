# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



%matplotlib inline
df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')
df_train.info(),

df_test.info()
df_train.isnull().sum()
df_train.dropna(how='any', inplace=True)

df_test.dropna(how='any', inplace=True)
df_train.head(3)
# Codificando texto em número

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        df_train[col] = df_train[col].astype('category').cat.codes
df_train.head(3)
df_trainc = df_train[['regiao', 'estado', 'municipio', 'gasto_pc_educacao', 'exp_anos_estudo']]
sns.heatmap(df_trainc.corr())
X = df_train[['regiao', 'estado', 'municipio', 'gasto_pc_educacao', 'exp_anos_estudo']]

y = df_train['nota_mat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

df.shape, test.shape
df = df.append(test, sort=False)
df.shape
df.head(3)
df.info()
df['codigo_mun'] = df['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))

df['codigo_mun'] = df['codigo_mun'].values.astype('int64')
df['nota_mat'] = np.log(df['nota_mat'])
df.shape
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
for c in df.columns:

    if df[c].dtype == 'object':

        df[c] = df[c].astype('category').cat.codes
df.min().min()
df.head(3)
df.info()
df.fillna(-2, inplace=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
df, test = df[df['nota_mat']!=-2], df[df['nota_mat']==-2]
train, valid = train_test_split(df, random_state=42)
rf = RandomForestRegressor(random_state=42, n_estimators=100)
feats = [c for c in df.columns if c not in ['nota_mat', 'indice_governanca', 'ranking_igm']]
rf.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error
valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
test['nota_mat'] = np.exp(rf.predict(test[feats]))
test.info()
test[['codigo_mun','nota_mat']].to_csv('submission.csv', index=False)