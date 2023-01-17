import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import PowerTransformer, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.svm import SVR

from sklearn.datasets import load_boston

from sklearn.ensemble import RandomForestRegressor
boston = load_boston()

dataset = pd.DataFrame(boston.data)

dataset.columns = boston.feature_names

dataset['MEDV'] = boston.target
dataset.head()
dataset.shape
dataset.describe()
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in dataset.items():

    ax = sns.boxplot(y=k, data=dataset, ax=axs[index])

    ax.set_title(dataset.columns[index] + " boxplot")

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in dataset.items():

    try:

        plot = sns.distplot(v, ax=axs[index])

        plot.set_title(dataset.columns[index] + " dist plot")

    except RuntimeError as re:

        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

            plot = sns.distplot(v, kde_kws={'bw': 0.1}, ax=axs[index])

            plot.set_title(dataset.columns[index] + " dist plot")

        else:

            raise re



    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.figure(figsize=(20, 10))

sns.heatmap(dataset.corr().abs(), annot=True)
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for i, k in enumerate(dataset.columns[:-1]):

    sns.regplot(y=dataset['MEDV'], x=dataset[k], ax=axs[i], color=np.random.rand(3,))

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
dataset = dataset[~(dataset['MEDV'] >= 50.0)]



X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
print('old skewness: ', y_train.skew())

pt_y = PowerTransformer(method='yeo-johnson', standardize=False)

y_train_ols = pt_y.fit_transform(y_train.values.reshape(len(y_train), 1))

y_test_ols = pt_y.transform(y_test.values.reshape(len(y_test), 1))

print('new skewness: ', stats.skew(y_train_ols)[0])
X_train_svr = X_train.loc[:, ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']]

X_test_svr = X_test.loc[:, ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']]
sc_X = StandardScaler()

sc_y = StandardScaler()

X_train_svr = sc_X.fit_transform(X_train_svr)

X_test_svr = sc_X.transform(X_test_svr)

y_train_svr = sc_y.fit_transform(y_train.values.reshape(len(y_train), 1))
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train_ols)



y_pred_ols = pt_y.inverse_transform(lin_reg.predict(X_test))



error_ols = mean_squared_error(pt_y.inverse_transform(y_test_ols), y_pred_ols, squared=False)

r2_ols = r2_score(pt_y.inverse_transform(y_test_ols), y_pred_ols)



print('RMSE: ', error_ols)

print('R2: ', r2_ols)
svr = SVR(kernel='rbf')

svr.fit(X_train_svr, y_train_svr.ravel())



y_pred_svr = sc_y.inverse_transform(svr.predict(X_test_svr))



error_svr = mean_squared_error(y_test, y_pred_svr, squared=False)

r2_svr = r2_score(y_test, y_pred_svr)



print('RMSE: ', error_svr)

print('R2: ', r2_svr)
tree_reg = RandomForestRegressor(n_estimators=20, random_state=0)

tree_reg.fit(X_train, y_train)



y_pred = tree_reg.predict(X_test)



error_tree = mean_squared_error(y_test, y_pred, squared=False)

r2_tree = r2_score(y_test, y_pred)



print('RMSE: ', error_tree)

print('R2: ', r2_tree)
pd.DataFrame([[error_ols, error_svr, error_tree], [r2_ols, r2_svr, r2_tree]], index=['RMSE', 'R2'], columns=['OLS reg', 'SVR', 'Random Forest reg'])