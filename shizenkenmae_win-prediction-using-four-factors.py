import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/college-basketball-dataset/cbb.csv")
df.info()
df.corr()['W'].sort_values()[:-1]
df_ff = df[['EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','W']]
fig, axes = plt.subplots(ncols=4, nrows=3)

for col, ax in zip(df_ff.columns, axes.flat):

    sns.distplot(df_ff[col], hist=False, ax=ax)

plt.tight_layout()

plt.show()
df_ff.describe()
sns.regplot(x='W',y = 'EFG_O',data = df_ff,scatter= True, fit_reg=True)
sns.regplot(x='W',y = 'TOR',data = df_ff,scatter= True, fit_reg=True)
sns.regplot(x='W',y = 'ORB',data = df_ff,scatter= True, fit_reg=True)
sns.regplot(x='W',y = 'FTR',data = df_ff,scatter= True, fit_reg=True)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

sns.regplot(x='W',y = 'EFG_D',data = df_ff,scatter= True, fit_reg=True,ax=ax1)

sns.regplot(x='W',y = 'TORD',data = df_ff,scatter= True, fit_reg=True,ax=ax2)

sns.regplot(x='W',y = 'DRB',data = df_ff,scatter= True, fit_reg=True,ax=ax3)

sns.regplot(x='W',y = 'FTRD',data = df_ff,scatter= True, fit_reg=True,ax=ax4)
# prepare the dataset

df_ff = df[['EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD']]

df_ff_y = df['W']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(df_ff, df_ff_y, test_size=0.25, random_state=21)

reg = LinearRegression()

reg = reg.fit(X_train,y_train)
print('Intercept: ', reg.intercept_)

print('R^2 score: ',reg.score(X_train,y_train))
coeff_df = pd.DataFrame(reg.coef_, df_ff.columns, columns=['Coefficient'])

coeff_df
cof = []

tcof = 0

for i in range(0,8,2):

    avgcof = (abs(coeff_df['Coefficient'][i])+abs(coeff_df['Coefficient'][i+1]))/2

    cof.append(avgcof)

    tcof += avgcof

print(cof/tcof)
# predict from the test set

y_pred = reg.predict(X_test)
# analyze the prediction

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
dfd = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'AbsDiff': abs(y_test-y_pred)})

dfd.sort_values(by=['AbsDiff'], inplace=True, ascending=True)

dfd[:10]
df[df.index == 21]
df[df.index == 937]
dfd.sort_values(by=['AbsDiff'], inplace=True, ascending=False)

dfd[:10]
df[df.index == 1174]
df[df.index == 1338]