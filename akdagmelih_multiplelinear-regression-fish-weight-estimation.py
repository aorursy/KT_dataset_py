# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fish-market/Fish.csv')

df = data.copy()

df.sample(10)
df.rename(columns= {'Length1':'LengthVer', 'Length2':'LengthDia', 'Length3':'LengthCro'}, inplace=True)

df.head()
df.info()
print(str('Is there any NaN value in the dataset: '), df.isnull().values.any())
sp = df['Species'].value_counts()

sp = pd.DataFrame(sp)

sp.T
sns.barplot(x=sp.index, y=sp['Species']);

plt.xlabel('Species')

plt.ylabel('Counts of Species')

plt.show()
df.corr()
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu');
g = sns.pairplot(df, kind='scatter', hue='Species');
df.describe().T
sns.boxplot(x=df['Weight']);
dfw = df['Weight']

dfw_Q1 = dfw.quantile(0.25)

dfw_Q3 = dfw.quantile(0.75)

dfw_IQR = dfw_Q3 - dfw_Q1

dfw_lowerend = dfw_Q1 - (1.5 * dfw_IQR)

dfw_upperend = dfw_Q3 + (1.5 * dfw_IQR)
dfw_outliers = dfw[(dfw < dfw_lowerend) | (dfw > dfw_upperend)]

dfw_outliers
sns.boxplot(x=df['LengthVer']);
dflv = df['LengthVer']

dflv_Q1 = dflv.quantile(0.25)

dflv_Q3 = dflv.quantile(0.75)

dflv_IQR = dflv_Q3 - dflv_Q1

dflv_lowerend = dflv_Q1 - (1.5 * dflv_IQR)

dflv_upperend = dflv_Q3 + (1.5 * dflv_IQR)



dflv_outliers = dflv[(dflv < dflv_lowerend) | (dflv > dflv_upperend)]

dflv_outliers
sns.boxplot(x=df['LengthDia']);
dfdia = df['LengthDia']

dfdia_Q1 = dfdia.quantile(0.25)

dfdia_Q3 = dfdia.quantile(0.75)

dfdia_IQR = dfdia_Q3 - dfdia_Q1

dfdia_lowerend = dfdia_Q1 - (1.5 * dfdia_IQR)

dfdia_upperend = dfdia_Q3 + (1.5 * dfdia_IQR)



dfdia_outliers = dfdia[(dfdia < dfdia_lowerend) | (dfdia > dfdia_upperend)]

dfdia_outliers
sns.boxplot(x=df['LengthCro']);
dfcro = df['LengthCro']

dfcro_Q1 = dfcro.quantile(0.25)

dfcro_Q3 = dfcro.quantile(0.75)

dfcro_IQR = dfcro_Q3 - dfcro_Q1

dfcro_lowerend = dfcro_Q1 - (1.5 * dfcro_IQR)

dfcro_upperend = dfcro_Q3 + (1.5 * dfcro_IQR)



dfcro_outliers = dfcro[(dfcro < dfcro_lowerend) | (dfcro > dfcro_upperend)]

dfcro_outliers
df[142:145]
df1 = df.drop([142,143,144])

df1.describe().T
# Dependant (Target) Variable:

y = df1['Weight']

# Independant Variables:

X = df1.iloc[:,2:7]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('How many samples do we have in our test and train datasets?')

print('X_train: ', np.shape(X_train))

print('y_train: ', np.shape(y_train))

print('X_test: ', np.shape(X_test))

print('y_test: ', np.shape(y_test))
reg = LinearRegression()

reg.fit(X_train,y_train)
# My model's parameters:

print('Model intercept: ', reg.intercept_)

print('Model coefficients: ', reg.coef_)
print('y = ' + str('%.2f' % reg.intercept_) + ' + ' + str('%.2f' % reg.coef_[0]) + '*X1 ' + str('%.2f' % reg.coef_[1]) + '*X2 ' +

      str('%.2f' % reg.coef_[2]) + '*X3 + ' + str('%.2f' % reg.coef_[3]) + '*X4 + ' + str('%.2f' % reg.coef_[4]) + '*X5')
y_head = reg.predict(X_train)
r2_score(y_train, y_head)
from sklearn.model_selection import cross_val_score

cross_val_score_train = cross_val_score(reg, X_train, y_train, cv=10, scoring='r2')

print(cross_val_score_train)
cross_val_score_train.mean()
y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))
plt.scatter(X_test['LengthCro'], y_test, color='red', alpha=0.4)

plt.scatter(X_test['LengthCro'], y_pred, color='blue', alpha=0.4)

plt.xlabel('Cross Length in cm')

plt.ylabel('Weight of the fish')

plt.title('Linear Regression Model for Weight Estimation');
plt.scatter(X_test['LengthVer'], y_test, color='purple', alpha=0.5)

plt.scatter(X_test['LengthVer'], y_pred, color='orange', alpha=0.5)

plt.xlabel('Vertical Length in cm')

plt.ylabel('Weight of the fish')

plt.title('Linear Regression Model for Weight Estimation');
plt.scatter(X_test['LengthDia'], y_test, color='purple', alpha=0.4)

plt.scatter(X_test['LengthDia'], y_pred, color='green', alpha=0.4)

plt.xlabel('Diagonal Length in cm')

plt.ylabel('Weight of the fish')

plt.title('Linear Regression Model for Weight Estimation');
plt.scatter(X_test['Height'], y_test, color='orange', alpha=0.5)

plt.scatter(X_test['Height'], y_pred, color='blue', alpha=0.5)

plt.xlabel('Height in cm')

plt.ylabel('Weight of the fish')

plt.title('Linear Regression Model for Weight Estimation');
plt.scatter(X_test['Width'], y_test, color='gray', alpha=0.5)

plt.scatter(X_test['Width'], y_pred, color='red', alpha=0.5)

plt.xlabel('Width in cm')

plt.ylabel('Weight of the fish')

plt.title('Linear Regression Model for Weight Estimation');
y_pred1 = pd.DataFrame(y_pred, columns=['Estimated Weight'])

y_pred1.head()
y_test1 = pd.DataFrame(y_test)

y_test1 = y_test1.reset_index(drop=True)

y_test1.head()
ynew = pd.concat([y_test1, y_pred1], axis=1)

ynew