import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.stattools import grangercausalitytests

import seaborn as sns

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.api import VAR

import math

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.neighbors import KNeighborsRegressor
%matplotlib inline
df = pd.read_csv('../input/into-the-future/train.csv')

df_test = pd.read_csv('../input/into-the-future/test.csv')
df.iloc[455]
df.tail()
pd.isna(df).sum()
df.info()
df['time'] = pd.to_datetime(df['time'])
df.info()
df.describe()
fig, ax0 = plt.subplots(figsize=(15, 5))

ax0.plot(df['feature_1'], 'r')

ax0.plot(df['feature_2'], 'b')
# Scaling data

scaler = MinMaxScaler()

df[['feature_1', 'feature_2']] = scaler.fit_transform(df[['feature_1', 'feature_2']])
df.head()
df_time = df.copy()

df_time.index = df['time']

fig, ax1 = plt.subplots(figsize=(15, 5))

ax1.plot(df_time['feature_1'], 'r')

ax1.plot(df_time['feature_2'], 'b')
df_data = df.drop(['id', 'time'], axis=1)

df_data.head()
corr = df_data.corr()

sns.heatmap(corr, vmax=-1.0, vmin=-0.6)
def is_GrangerCause(data=None, maxlag=5):

    """This function find if x2 Granger cause x1 vis versa """    

    gc = grangercausalitytests(data, maxlag=maxlag, verbose=False)

    

    for i in range(maxlag):

        x=gc[i+1][0]

        p1 = x['lrtest'][1] # pvalue for lr test

        p2 = x['ssr_ftest'][1] # pvalue for ssr ftest

        p3 = x['ssr_chi2test'][1] #pvalue for ssr_chi2test

        p4 = x['params_ftest'][1] #pvalue for 'params_ftest'

        

        condition = ((p1 < 0.05 and p2 < 0.05) and (p3 < 0.05 and p4 < 0.05))

        

        if condition == True:

            cols = data.columns

            print('Yes: {} Granger causes {}'.format(cols[0], cols[1]))

            print('maxlag = {}\nResults: {}'.format(i, x))

            break

            

        else:

            if i == maxlag - 1:

                cols = data.columns

                print('No: {} does not Granger cause {}'.format(cols[0], cols[1]))
is_GrangerCause(df_data)
def adfuller_test(data, p_val=0.05):

    print('\n \n ----------------- ADF-------------------')

    index = ['Test Stats', 'p-value', 'Lags', 'Observations']

    adf_test = adfuller(data, autolag='AIC')

    adf = pd.Series(adf_test[0:4], index =index)

    

    for key, value in adf_test[4].items():

        adf['Critical Value(%s)'%key] = value

    print(adf)

    

    p = adf['p-value']

    if p <= p_val:

        print('Series is stationary')

    else:

        print('Series is not stationary')
# Applying test on both series

adfuller_test(df_data['feature_1'])

adfuller_test(df_data['feature_2'])
df_sup = pd.read_csv('../input/into-the-future/train.csv')

df_sup = df_sup.drop(['id', 'time',], axis=1)
df_sup.head()
# Scaling

scaler = StandardScaler()

df_sup[['feature_1', 'feature_2']] = scaler.fit_transform(df_sup[['feature_1', 'feature_2']])
df_sup.head()
df_sup.describe()
df_test = pd.read_csv('../input/into-the-future/test.csv')
df_test = df_test.drop('time', axis=1)

id_test = df_test['id']

feature_1_test = df_test['feature_1']
df_test.head()
df_test[['feature_1', 'id']] = scaler.transform(df_test[['feature_1', 'id']])
df_test.head()
df_test['feature_1'].plot()
# Comparing df_test to df_sup

fig, ax1 = plt.subplots(nrows=2, sharex= True, figsize=(15, 5))

ax1[0].plot(df_sup['feature_1'], 'r')

ax1[1].plot(df_test['feature_1'], 'b')
sns.relplot(x='feature_1', y='feature_2', data=df_sup, kind='scatter')
df_sup['feature_1'] = df_sup[df_sup['feature_1'] < 3]

df_sup = df_sup.dropna()
df_sup.describe()
sns.regplot(x='feature_1', y='feature_2', data=df_sup)
x_train,x_valid, y_train,  y_valid = train_test_split(df_sup['feature_1'], df_sup['feature_2'], test_size=0.2)
x_train, y_train = pd.DataFrame(x_train, columns=['feature_1']), pd.DataFrame(y_train, columns=['feature_2'])

x_valid, y_valid = pd.DataFrame(x_valid, columns=['feature_1']), pd.DataFrame(y_valid, columns=['feature_2'])
Lr = LinearRegression()
Lr.fit(x_train, y_train)
pred_valid = Lr.predict(x_valid)
knn = KNeighborsRegressor()
knn_param = {'n_neighbors': [1, 3, 5],

            'leaf_size': [10, 20, 40, 50]}
grid_search = GridSearchCV(knn, knn_param, n_jobs=-1, cv=10)
grid_search.fit(x_train, y_train)
grid_search.best_score_
grid_search.best_params_
model = grid_search.best_estimator_
pred_valid = model.predict(x_valid)
r2_score(y_valid, pred_valid)
plt.figure(figsize=(10, 10))

plt.scatter(x_train, y_train, color = "red")

plt.plot(x_train, model.predict(x_train), 'go')

plt.title("Knn Fit")

plt.xlabel("feature_1")

plt.ylabel("feature_2")

plt.show()
x_test = pd.DataFrame(df_test['feature_1'], columns=['feature_1'])
pred_test = pd.DataFrame(model.predict(x_test), columns=['feature_2'])
pred_test
pred_test_df = pd.concat([df_test['id'], pred_test], axis=1, copy=False)
pred_test_df
pred_test_df[['id', 'feature_2']] = scaler.inverse_transform(pred_test_df[['id', 'feature_2']])
pred_test_df['id'] = id_test
pred_test_df.shape
pred_test_df.head()
len(id_test)
len(pred_test_df['feature_2'])
pred_test_df['feature_2']
submission = pd.DataFrame({'id': id_test, 'feature_2':pred_test_df['feature_2'].values})
submission.head()
submission.to_csv(r'submisson.csv', index=False)
# adding timestamp

df_data.index = pd.to_datetime(df['time'])

df_data.head()
model = VAR(df_data, freq='10S')
result = model.fit(maxlags=30, ic='fpe')

result.summary()
# Forecasting

lag_order = result.k_ar

result.forecast(df_data.values[-lag_order:], 6)
result.plot_forecast(375)
# Evaluation

fevd = result.fevd(5)

fevd.summary()
test_predicted = result.forecast(df_data.values[-lag_order:], 375)

test_predicted_df = pd.DataFrame(test_predicted, index=df_test['id'], columns=df_data.columns)
test_predicted_df[['feature_1', 'feature_2']] = scaler.inverse_transform(test_predicted_df[['feature_1', 'feature_2']])
test_predicted_df.head()
df_test.index = df_test['id']

df_test.head()
math.sqrt(mean_squared_error(df_test['feature_1'], test_predicted_df['feature_1']))
# Plotting data simulatneously to visualize quality of prediction

fig, ax3 = plt.subplots(figsize=(15, 5))

ax3.plot(df_test['feature_1'], 'r')

ax3.plot(test_predicted_df['feature_1'], 'b')