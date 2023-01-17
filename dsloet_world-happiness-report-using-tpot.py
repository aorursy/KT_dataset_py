# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/2015.csv')
df.head()
df.shape
plt.scatter(np.sort(df['Economy (GDP per Capita)'].values), df['Happiness Score'])
plt.scatter(np.sort(df['Health (Life Expectancy)'].values), df['Happiness Score'])
plt.scatter(np.sort(df['Freedom'].values), df['Happiness Score'])
plt.scatter(np.sort(df['Freedom'].values), df['Happiness Score'])
plt.scatter(np.sort(df['Trust (Government Corruption)'].values), df['Happiness Score'])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(df['Country'])

Country1 = pd.DataFrame({'Country1': le.transform(df['Country'])})

df = pd.concat([df, Country1], axis=1)
corrmat = df.corr()

plt.subplots(figsize=(4,4))

sns.heatmap(corrmat, vmax=0.9, square=True)
plt.figure(figsize=(16,9))

sns.heatmap(df.corr()[['Happiness Score']].sort_values('Happiness Score'), annot=True)
plt.scatter(df['Happiness Score'], df['Economy (GDP per Capita)'])

plt.show()
sns.regplot(x = df['Happiness Score'], y = df['Economy (GDP per Capita)'])
sns.regplot(x = df['Happiness Score'], y = df['Family'])
sns.regplot(x = df['Happiness Score'], y = df['Country1'])
sns.regplot(x = df['Happiness Score'], y = df['Standard Error'])
sns.regplot(x = df['Happiness Score'], y = df['Happiness Rank'])
sns.boxplot(df['Happiness Score'], df['Region'] )
le.fit(df['Region'])

Region1 = pd.DataFrame({'Region1': le.transform(df['Region'])})

df = pd.concat([df, Region1], axis=1)
plt.figure(figsize=(16,9))

sns.heatmap(df.corr()[['Happiness Score']].sort_values('Happiness Score'), annot=True)
sns.regplot(x = df['Region1'], y = df['Happiness Score'])
df.head()
X = df.as_matrix([['Region1', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom',

                  'Trust (Government Corruption)', 'Generosity']])

y = df['Happiness Score'] #regression job

#y2 = df['Country1'] #classification job


print(y.shape)

from tpot import TPOTRegressor, TPOTClassifier
TPOTReg = TPOTRegressor(generations=10, population_size=50, verbosity=2)
TPOTReg.fit(X, y)
TPOTReg.score(X, y)
y_pred = TPOTReg.predict(X)
plt.plot(y, label='True')

plt.plot(y_pred, label='pred')

plt.legend()

plt.show()
dft = pd.read_csv('../input/2016.csv')
le.fit(dft['Region'])

Region1 = pd.DataFrame({'Region1': le.transform(dft['Region'])})

df = pd.concat([dft, Region1], axis=1)
X_val = dft.as_matrix([['Region1', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom',

                  'Trust (Government Corruption)', 'Generosity']])

y_val = dft['Happiness Score']
y_val_pred = TPOTReg.predict(X_val)
y_val_pred
y_val.ravel(157,)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_val_pred)
plt.plot(y_val, label='True')

plt.plot(y_val_pred, label='pred')

plt.legend()

plt.show()