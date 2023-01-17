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
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv').fillna(value=np.nan)

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv').fillna(value=np.nan)

sub = pd.DataFrame()

sub['Id'] = df_test['Id']

sub['SalePrice'] = pd.Series()



for col in df:

    dtype = str(df[col].dtype)

    uniq = str(pd.unique(df[col]).size)

    print(col + " " + dtype + " " + uniq)
df["GarageYrBlt"].head()
import seaborn as sns

import matplotlib.pyplot as plt



corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
categoric = ['OverallQual','GarageCars']

continous = []

corr_limit = 0.2

corr_high = 0.6

corr = corrmat["SalePrice"]

lowvals = []

highvals = []

for idx, val in corr.iteritems():

    if val < corr_limit:

        lowvals.append(idx)

    if val > corr_high:

        highvals.append(idx)

lowvals.remove("Id")

for val in highvals:

    if val not in categoric:

        continous.append(val)



highvals
df = df.drop(lowvals, axis=1)

df_test = df_test.drop(lowvals, axis=1)

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#todo: 

#remove flyers

#find categoricals to 1hotenc

#normalize values

#build model
for var in continous:

    data = pd.concat([df['SalePrice'], df[var]], axis=1)

    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
for var in categoric:

    data = pd.concat([df['SalePrice'], df[var]], axis=1)

    f, ax = plt.subplots(figsize=(8, 6))

    fig = sns.boxplot(x=var, y="SalePrice", data=data)

    fig.axis(ymin=0, ymax=800000);
sns.set()

#sns.pairplot(df[highvals], size = 2.5)

plt.show();
df = df[highvals]



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)

#df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#df_train.isnull().sum().max()

df.head()
from scipy import stats

from scipy.stats import norm

sns.distplot(df['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['SalePrice'], plot=plt)
sample_transformed, lambd = stats.boxcox(df['SalePrice'])

sns.distplot(sample_transformed, fit=norm);

fig = plt.figure()

res = stats.probplot(sample_transformed, plot=plt)
sample_transformed = np.log(sample_transformed)

sns.distplot(sample_transformed, fit=norm);

fig = plt.figure()

res = stats.probplot(sample_transformed, plot=plt)
sample_transformed = np.exp(sample_transformed)

sns.distplot(sample_transformed, fit=norm);

fig = plt.figure()

res = stats.probplot(sample_transformed, plot=plt)

highvals
from sklearn import preprocessing



highvals.remove('SalePrice')

mm_scaler = preprocessing.MinMaxScaler()

mm_scaler.fit(df[highvals])

df[highvals] = mm_scaler.transform(df[highvals])

df_test = mm_scaler.transform(df_test[highvals])



x_train = df[highvals]





y_train = pd.DataFrame()

price_scaled = []

for price in df['SalePrice']:

    price_scaled.append(price/max(df['SalePrice']))

y_train['SalePrice'] = price_scaled

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
df_test
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



model = Sequential()

model.add(Dense(12, input_dim=6, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='linear'))



opt = Adam(lr=1e-3, decay=1e-3 / 200)

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(x_train, y_train, epochs=150, batch_size=10)
print(history.history.keys())

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
y_train
df_test = pd.DataFrame(df_test).fillna(value=0)

target = model.predict(df_test)

target_scaled = []

for val in np.reshape(target,len(target)):

    target_scaled.append(max(df['SalePrice']) * val)

sub['SalePrice'] = pd.Series(target_scaled)

sub
sub.to_csv('submission.csv', index = False)
pd.DataFrame(df_test)[:661]