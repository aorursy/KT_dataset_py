# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pydot

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from tensorflow import keras

from keras import layers

from keras import models

from sklearn.preprocessing import LabelEncoder

from keras import regularizers

from keras.metrics import mean_squared_logarithmic_error

from sklearn.model_selection import KFold

from keras import optimizers

from sklearn.preprocessing import normalize

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/houseprices/HousePrice.csv')
df.head()
df.shape
df.describe()
sns.jointplot(x='n_hot_rooms', y='price', data=df)
sns.jointplot(x='rainfall', y='price', data=df)
df.head(5)
sns.countplot(x='airport', data=df)
sns.countplot(x='waterbody', data=df)
sns.countplot(x='bus_ter', data=df)
df.info()
np.percentile(df.n_hot_rooms,[99])
np.percentile(df.n_hot_rooms,[99])[0]
nv = np.percentile(df.n_hot_rooms,[99])[0]
df[(df.n_hot_rooms > nv)]
df.n_hot_rooms[(df.n_hot_rooms > 3 * nv)] = 3 * nv
df[(df.n_hot_rooms > nv)]
np.percentile(df.rainfall,[1])[0]
lv = np.percentile(df.rainfall,[1])[0]
df[(df.rainfall < lv)]
df.rainfall[(df.rainfall < 0.3 * lv)] = 0.3 * lv
df[(df.rainfall < lv)]
#sns.jointplot(x="crime_rate", y="price", data=df)
df.info()
#Impute Missing values for 1 columns

df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())

# For all columns : df = df.fillna(df.mean())
df.info()
df.head()
df['avg_dist'] = (df.dist1 + df.dist2 + df.dist3 + df.dist4) / 4
df.describe()
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
df.head()
del df['bus_ter']
df.head()
df = pd.get_dummies(df)
df.head()
del df['airport_NO']
del df['waterbody_None']
df.head()
df.corr()
del df['parks']
df.head()
import statsmodels.api as sn
X = sn.add_constant(df['room_num'])
lin_model = sn.OLS(df['price'], X).fit()
lin_model.summary()
y = df['price']
X = df[['room_num']]
lin_model2 = LinearRegression()
# Fit the Model

lin_model2.fit(X,y)
print(lin_model2.intercept_, lin_model2.coef_)
#help(lin_model2)
# Make the Price Prediction

lin_model2.predict(X)
pricepredic = lin_model2.predict(X)

Submission = pd.DataFrame({"Predicted Price" : pricepredic}).to_csv("submission_SLR.csv")
submission = pd.read_csv('submission_SLR.csv', index_col=0)

submission.head(15)
# plot de prediction 

sns.jointplot(x = df['room_num'], y = df['price'], data = df, kind = "reg")
df.head()
X_multi = df.drop("price", axis = 1)
X_multi.head(2)
y_multi = df['price']
y_multi
X_multi_cons = sn.add_constant(X_multi)
X_multi_cons.head()
# Create Model

lm_multi = sn.OLS(y_multi, X_multi_cons).fit()
lm_multi.summary()
lin_model3 = LinearRegression()
lin_model3.fit(X_multi, y_multi)
print(lin_model3.intercept_, lin_model3.coef_)
lin_model3.predict(X_multi)
pricepredic2 = lin_model3.predict(X_multi)

Submission2 = pd.DataFrame({"Predicted Price" : pricepredic}).to_csv("submission_MLR.csv")
submission2 = pd.read_csv('submission_MLR.csv', index_col=0)

submission.head(15)