import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
insure_data = pd.read_csv('../input/caravan-insurance-challenge.csv')
insure_data.head()
insure_data.describe()
df = insure_data[['MINKGEM','MZPART','MRELGE','MOPLHOOG','MHKOOP','MAUT0']]

df.head()
df.columns = ['Average Income','Percent Prvte Hlth Insure','Percent Married','Percent High Education',

              'Percent Home Owners','Percent No Car']
df.head()
income = insure_data[['MINKM30','MINK3045','MINK4575','MINK7512','MINK123M','MINKGEM']]
income.columns = ["< 30,000", "30,000 - 45,000", '45,000 - 75,000', '75,000 - 120,000', '> 120,000', 'Percent Near Average']
def data_conversion(num):

    percent_dict = {0 : 0, 1 : .05, 2 : .17, 3 : .30, 4 : .43,

                    5 : .56, 6 : .69, 7 : .84, 8 : .94, 9 : 1.0}

    return percent_dict[num]
income = income.applymap(data_conversion)
income.loc[:,'< 30,000'] *= 15000

income.loc[:,'30,000 - 45,000'] *= 37500

income.loc[:, '45,000 - 75,000'] *= 60000

income.loc[:, '75,000 - 120,000'] *= 97500

income.loc[:,'> 120,000'] *= 120000
income['Average Income'] = income.sum(axis = 1)
income.head()
df.loc[:,'Average Income'] = income['Average Income']

df.head()
sns.set_style('ticks')

sns.jointplot(x = 'Percent High Education', y = 'Average Income', data = df, kind='kde')
sns.jointplot(y = 'Percent Home Owners', x = 'Average Income', data = df, kind='kde')
sns.jointplot(y = 'Percent No Car', x = 'Average Income', data = df, kind='kde')
sns.jointplot(y = 'Percent Prvte Hlth Insure', x = 'Average Income', data = df, kind='kde')
sns.jointplot(y = 'Percent Prvte Hlth Insure', x = 'Percent No Car', data = df, kind='kde')
sns.jointplot(x = 'Average Income', y = 'Percent Married', data = df, kind='kde')
sns.jointplot(y = 'Percent Prvte Hlth Insure', x = 'Percent High Education', data = df, kind='kde')
sns.jointplot(x = 'Percent Home Owners', y = 'Percent Prvte Hlth Insure', data = df, kind='kde')
from sklearn.model_selection import train_test_split
X = df[['Average Income','Percent High Education','Percent Married','Percent No Car','Percent Home Owners']]

Y = df['Percent Prvte Hlth Insure']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2017)
from sklearn.linear_model import LinearRegression
linearmodel = LinearRegression()

linearmodel.fit(X_train, Y_train)
linear_predict = linearmodel.predict(X_test)
from sklearn import metrics
metrics.mean_absolute_error(Y_test, linear_predict)
metrics.mean_squared_error(Y_test, linear_predict)
np.sqrt(metrics.mean_squared_error(Y_test, linear_predict))
plt.scatter(Y_test, linear_predict)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
dtree_predict = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test, dtree_predict))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500)

rfc.fit(X_train, Y_train)
RndForPred = rfc.predict(X_test)
print(classification_report(Y_test, RndForPred))