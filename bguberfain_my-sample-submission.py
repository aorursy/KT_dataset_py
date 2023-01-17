import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



%matplotlib inline
df_train = pd.read_csv('../input/train.csv', index_col='id')

df_test = pd.read_csv('../input/test.csv', index_col='id')
df_train.head()
df_train['vlrLiquido'].hist(log=True, bins=21)
n_train = len(df_train)

df_both = pd.concat([

    df_train.drop(['vlrLiquido'], axis=1),

    df_test

])
dummy_cols = ['txNomeParlamentar', 'numMes', 'sgUF', 'sgPartido', 'txtDescricao']
X = pd.get_dummies(df_both[dummy_cols]).values
X_train = X[:n_train]

y_train = np.log1p(df_train['vlrLiquido'])

X_test = X[n_train:]



# Remove nan do treino

X_train = X_train[np.isfinite(y_train)]

y_train = y_train[np.isfinite(y_train)]
model = LinearRegression().fit(X_train, y_train)
y_pred_train = model.predict(X_train)

plt.figure(figsize=(10, 10))

plt.scatter(y_train, y_pred_train, s=1, alpha=0.5)

plt.xlim(0, 12)

plt.ylim(0, 12)

plt.gca().set_aspect('equal', adjustable='box')
df_test['vlrLiquido'] = np.expm1(model.predict(X_test).clip(0, 12)).round(2)
df_test[['vlrLiquido']].to_csv('submission.csv')
!head submission.csv