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

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn import preprocessing

from sklearn import metrics

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from sklearn.metrics import r2_score

!pip install catboost

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import xgboost as xgb

!pip install xgboost
df = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
df.head()
fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
del df['url']

del df['region_url']

del df['vin']

del df['title_status']

del df['size']

del df['image_url']

del df['county']

del df['id']

del df['state']

del df['long']

del df['lat']

del df['description']

del df['region']
fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
df = df.dropna()
df.drop(df[df.price.values == 0].index, inplace = True)

df.drop(df[df.year.values > 2019].index, inplace = True)

df.drop(df[df.odometer.values > 55000].index, inplace = True)

df.price.dropna(axis = 0, inplace = True)

df.drop(df[df.price.values < 700].index, inplace = True)
df['year'] = (df['year']-1900).astype(int)

df['odometer'] = df['odometer'].astype(int)
df = df[df['price'] > 1000]

df = df[df['price'] < 40000]

# Rounded ['odometer'] to 5000

df['odometer'] = df['odometer'] // 5000

df = df[df['year'] > 110]
import string

# realization preprocessing

def preprocess(doc):

    try:

        # lower the text

        doc = doc.lower()

        # remove punctuation, spaces, etc.

        for p in string.punctuation + string.whitespace:

            doc = doc.replace(p, ' ')

        # remove extra spaces, merge back

        doc = doc.strip()

        doc = ' '.join([w for w in doc.split(' ') if w != ''])

    except:

        pass

    return doc
for colname in df.select_dtypes(include = np.object).columns:

    df[colname] = df[colname].map(preprocess)

df.head()
df = df[:50000]
columns = ['manufacturer', 'fuel', 'type', 'transmission', 'drive', 'paint_color', 'model', 'cylinders', 'condition']
le = LabelEncoder()

for col in columns:

    if col in df.columns:

        le.fit(list(df[col].astype(str).values))

        df[col] = le.transform(list(df[col].astype(str).values))
scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df.corr()
# sample data for best results

df = df.sample(frac=1).reset_index(drop=True)
y = df['price']

y
del df['price']
X_classic = df
X_classic
X_train_classic, X_test_classic, y_train, y_test = train_test_split(X_classic, y, test_size=0.20)
X_train_classic.shape, X_test_classic.shape
%%time

reg = LinearRegression().fit(X_train_classic, y_train)
predictions = reg.predict(X_test_classic)
print(metrics.r2_score(y_test, predictions))
print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
%%time

alphas = np.linspace(1, 1000, 100)



weights = np.empty((len(X_classic.columns), 0))

for alpha in alphas:

    ridge_regressor = Ridge(alpha)

    ridge_regressor.fit(X_train_classic, y_train)

    weights = np.hstack((weights, ridge_regressor.coef_.reshape(-1, 1)))

plt.plot(alphas, weights.T)

plt.xlabel('regularization coef')

plt.ylabel('weight value')

plt.show()
ridge = Ridge(alpha = 1)

ridge.fit(X_train_classic, y_train)

predictions = ridge.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
lasso = Lasso(alpha = 1)

lasso.fit(X_train_classic, y_train)

predictions = lasso.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
SVR = SVR()

SVR.fit(X_train_classic, y_train)

predictions = SVR.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
sgd = SGDRegressor(alpha = 0.002004008016032064, penalty = 'l2')

sgd.fit(X_train_classic, y_train)

predictions = sgd.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
decision_tree = DecisionTreeRegressor()

decision_tree.fit(X_train_classic, y_train)

predictions = decision_tree.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
rnd_forest_reg = RandomForestRegressor(max_depth = 14, min_samples_split = 2, n_estimators = 1000)

rnd_forest_reg.fit(X_train_classic, y_train)

predictions = rnd_forest_reg.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
Ada_Boost = AdaBoostRegressor()

Ada_Boost.fit(X_train_classic, y_train)

predictions = Ada_Boost.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
xbg_reg = xgb.XGBRegressor(max_depth = 7, learning_rate = 0.1, n_estimators = 130, reg_lambda = 0.5)

xbg_reg.fit(X_train_classic, y_train)

predictions = xbg_reg.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))
from catboost import CatBoostRegressor, Pool
model = CatBoostRegressor(iterations=15000, 

                           task_type="GPU",

                           devices='0:1')

model.fit(X_train_classic,

          y_train,

          verbose=False)



predictions = model.predict(X_test_classic)

print("MAE:", metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R2: ', metrics.r2_score(y_test, predictions))