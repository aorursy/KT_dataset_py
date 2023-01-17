# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
m_df = pd.read_csv('../input/boston.csv',low_memory=False)
m_df.head()
m_df.info()
m_df.describe()
m_df.corr()
Lst = ['PTRATIO', 'TAX', 'RAD', 'DIS', 'AGE', 'RM', 'NOX', 'INDUS', 'ZN', 'CRIM', 'LON', 'TRACT']
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(25,20)) 

sns.set(font_scale=1.5)

sns.heatmap(m_df.corr(),cbar=True,fmt =' .2f', annot=True, cmap='coolwarm')
#removed RAD becuase of high correlation between TAX and RAD

# removed NOX because of high correlction between NOX and INDUS

Lst = ['PTRATIO', 'TAX', 'DIS', 'AGE', 'RM', 'INDUS', 'ZN', 'CRIM', 'LON', 'TRACT', 'MEDV']
sns.pairplot(m_df[Lst])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()
y = sc_y.fit_transform(m_df.MEDV.values.reshape((506,1)))
m_df.drop('MEDV',axis=1,inplace=True)
m_df.drop('TOWN',axis=1,inplace=True)
m_df.info()
X = sc_X.fit_transform(m_df.as_matrix())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)
pred = lr.predict(X_test)
plt.scatter(sc_y.inverse_transform(y_test), sc_y.inverse_transform(pred))
sc_y.inverse_transform(y_test[4])
sc_y.inverse_transform(pred[4])
sc_y.inverse_transform(y_test[10])
sc_y.inverse_transform(pred[10])
from sklearn.metrics import r2_score
r2_score(y_test, pred)
r2_score(y_train, lr.predict(X_train))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train, y_train)
pred2 = rf.predict(X_test).reshape((152,1))
plt.scatter(sc_y.inverse_transform(y_test), sc_y.inverse_transform(pred2))
r2_score(y_test, pred2)
r2_score(y_train, rf.predict(X_train))