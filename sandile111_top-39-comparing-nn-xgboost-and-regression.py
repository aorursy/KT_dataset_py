# importing necessary dependencies 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns

import time



%matplotlib inline

%xmode plain



sns.set()
# Loading both test and train data

df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df.shape ,df_test.shape
df.columns
df.head()
df.tail()
fig = plt.figure(figsize=(16,9))



ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222, sharey=ax1)

ax3 = fig.add_subplot(223, sharey=ax1)

ax4 = fig.add_subplot(224, sharey=ax1)

ax1.scatter(df['LandContour'], df['SalePrice'])

ax1.set_xlabel('LandContour')

ax2.scatter(df['OverallCond'], df['SalePrice'])

ax2.set_xlabel('OverallCond')

ax3.scatter(df['YearBuilt'], df['SalePrice'])

ax3.set_xlabel('YearBuilt')

ax4.scatter(df['GrLivArea'], df['SalePrice'])

ax4.set_xlabel('GrLivArea')
fig = plt.figure(figsize=(16,9))



ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222, sharey=ax1)

ax3 = fig.add_subplot(223, sharey=ax1)

ax4 = fig.add_subplot(224, sharey=ax1)

ax1.scatter(df['BedroomAbvGr'], df['SalePrice'])

ax1.set_xlabel('BedroomAbvGr')

ax2.scatter(df['LotArea'], df['SalePrice'])

ax2.set_xlabel('LotArea')

ax3.scatter(df['GarageArea'], df['SalePrice'])

ax3.set_xlabel('GarageArea')

ax4.scatter(df['1stFlrSF'], df['SalePrice'])

ax4.set_xlabel('1stFlrSF')
# Plotting correlation heatmap with the help of Seaborn

_, figcorr = plt.subplots(figsize = (9,9))

sns.heatmap(df.corr(), ax = figcorr, center = 0)
k = 10

corr = df.corr()

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)



sns.set(font_scale=1.25)



hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}

                , yticklabels=cols.values, xticklabels=cols.values)

sns.pairplot(df[cols[:]], size=2.5)
# we combine the two data frames for the train and test data

# we split them later again, this makes the data preprocessing easier

y_train = df['SalePrice'].values

df = df.drop(columns=['SalePrice'])

dfc = pd.concat((df, df_test))

dfc.head()
total = dfc.isnull().sum().sort_values(ascending=False)

percent = (dfc.isnull().sum()/dfc.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(20)
dfc = dfc.drop(columns=missing_data[missing_data['Percent'] > 0.1 ].index.values, axis=1)
dfc.head()
dfc = dfc.fillna(method='bfill')
cols
ids = dfc['Id']

dfc = dfc.drop(columns = ['Id'])

dfc = pd.get_dummies(dfc)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

dfc = pd.DataFrame(np.log1p(dfc.values), columns=dfc.columns )

# dfc = pd.DataFrame(sc.fit_transform(dfc.values), columns=dfc.columns )
x_train = dfc.iloc[0:df.shape[0], :].values

x_validation = dfc.iloc[df.shape[0]:, :].values



x_train, x_test , y_train, y_test = train_test_split(x_train, y_train,

                                                         test_size = 0.30,

                                                        random_state = 45

                                                        )

print(x_train.shape, x_validation.shape)
y_train = np.log1p(y_train.reshape(-1,1))

y_test = np.log1p(y_test.reshape(-1,1))
# importing dependencies

import tensorflow.keras as keras

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from eli5.sklearn import PermutationImportance

import eli5

import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Defining Neural Network Model Structure



def make_model():

    

    model = keras.models.Sequential()



    model.add(keras.layers.Dense(units = 64 ,activation='relu', input_shape=x_train.shape[1:]))

    model.add(keras.layers.BatchNormalization())

    

    model.add(keras.layers.Dense(units = 32, activation='relu'))

    model.add(keras.layers.BatchNormalization())



    model.add(keras.layers.Dense(units = 16, activation='relu'))

    model.add(keras.layers.BatchNormalization())



    model.add(keras.layers.Dense(units = 1))

    

    optim = keras.optimizers.SGD(lr = 0.001)

    model.compile(loss='mse', optimizer=optim, metrics=['mse'])

    

    return model

callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss',

                                  min_delta = 0,

                                  patience = 5,

                                  verbose = 0, 

                                  mode='auto')]



my_model = KerasRegressor(build_fn=make_model, verbose=0)

my_model.fit(x_train, y_train,  epochs = 1000 , verbose=0, validation_data=(x_test, y_test), callbacks = callbacks)
perm = PermutationImportance(my_model, random_state=1).fit(x_train,y_train, verbose=0)

eli5.show_weights(perm, feature_names = dfc.columns.tolist())
# sort the sale prices in ascending order and plot their respective predictions



y_pred = my_model.predict(x_test)



args = np.argsort(y_test.squeeze(1))



out  = [y_test.squeeze(1)[i] for i in args]

y_out = [y_pred[i] for i in args]



plt.plot(np.arange(len(args)), y_out, label='prediction')

plt.plot(np.arange(len(args)), out,  label='sale price')

plt.legend(loc='upper left')
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import plot_importance





xgb = XGBRegressor(n=1000)

xgb.fit(x_train, y_train, verbose=False)
# sort the sale prices in ascending order and plot their respective predictions



y_pred = xgb.predict(x_test)



args = np.argsort(y_test.squeeze(1))



out  = [y_test.squeeze(1)[i] for i in args]

y_out = [y_pred[i] for i in args]



plt.plot(np.arange(len(args)), y_out, label='prediction')

plt.plot(np.arange(len(args)), out,  label='sale price')

plt.legend(loc='upper left')
plot_importance(xgb, max_num_features=5)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge, RidgeCV,  LassoCV
lg = LinearRegression()

lg.fit(x_train, y_train)
# sort the sale prices in ascending order and plot their respective predictions



y_pred = lg.predict(x_test)

args = np.argsort(y_test.squeeze(1))

out  = [y_test.squeeze(1)[i] for i in args]

y_out = [y_pred[i] for i in args]



plt.plot(np.arange(len(args)), y_out, label='prediction')

plt.plot(np.arange(len(args)), out,  label='sale price')

plt.legend(loc='upper left')
maxcoef = np.argsort(-np.abs(lg.coef_)).squeeze()



coef = lg.coef_[0][maxcoef-1]

for i in range(0, 5):

    print("{:.<025} {:< 010.4e}".format(dfc.columns[maxcoef[i]], coef[i]))
# Fitting a Lasso Regression Model



ls = LassoCV()

ls.fit(x_train, y_train.squeeze())
# sort the sale prices in ascending order and plot their respective predictions



y_pred = ls.predict(x_test)

args = np.argsort(y_test.squeeze(1))

out  = [y_test.squeeze(1)[i] for i in args]

y_out = [y_pred[i] for i in args]



plt.plot(np.arange(len(args)), y_out, label='prediction')

plt.plot(np.arange(len(args)), out,  label='sale price')

plt.legend(loc='upper left')
maxcoef = np.argsort(-np.abs(ls.coef_))

coef = ls.coef_[maxcoef]



for i in range(0, 5):

    print("{:.<025} {:< 010.4e}".format(dfc.columns[maxcoef[i]], coef[i]))

# fiting a Ridge Regression Model



rg = RidgeCV()

rg.fit(x_train, y_train)
# sort the sale prices in ascending order and plot their respective predictions



y_pred = rg.predict(x_test)

args = np.argsort(y_test.squeeze(1))

out  = [y_test.squeeze(1)[i] for i in args]

y_out = [y_pred[i] for i in args]



plt.plot(np.arange(len(args)), y_out, label='prediction')

plt.plot(np.arange(len(args)), out,  label='sale price')

plt.legend(loc='upper left')
maxcoef = np.argsort(-np.abs(rg.coef_).squeeze())

coef = rg.coef_.squeeze()[maxcoef]

for i in range(0, 5):

    print("{:.<025} {:< 010.4e}".format(dfc.columns[maxcoef[i]], coef[i]))
'''we're exponentiating the prediction to reverse the log transformation applied

 to the target during training'''

y_pred_ls = np.exp(ls.predict(x_validation)) 

y_pred_lg = np.exp(rg.predict(x_validation).squeeze())

print(y_pred_lg)

y_pred_ave = (y_pred_ls + y_pred_lg)/2



submit = pd.DataFrame({'Id': ids[df.shape[0]:].values, 'SalePrice':y_pred_ave})

submit.to_csv('submission'+str(time.time())+'.csv', index=False)

submit.head(10)
