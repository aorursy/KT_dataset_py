import numpy as np

import pandas as pd

import seaborn as sb
df = pd.read_csv("./pandas/pupils.csv")
df.head()
df.info()
df.describe()
df.avg.std()
df.describe()['income']['min']
sb.distplot(df.income)
sb.pairplot(df,hue='gen')
df.corr()
df.corr()['Height']['Weight'] > 0.9
df[df.Age > 10]
df[(df.Age > 10) & (df.income > 30000)]
df.query("Age > 10 and income > 30000")
df.query("Country in ('FR','GR')")
df.groupby(['Country']).mean()
df.groupby(['Country','gen']).mean()
df.head()
df.Age -= 10
df.drop([1,3,5],inplace=True)
df.drop(['Name','type'],axis=1,inplace=True)
df['inc_per_f'] = df.income / df.family
df.head()
ages = df.Age.values
type(ages)
ages-=10
sb.distplot(ages)
import scipy.stats as st
f = st.gaussian_kde(ages)
x = np.linspace(2,18,40)
import matplotlib.pyplot as plt
plt.plot(x , f(x))
f(8)
arr = df.income.values
sb.distplot(df.income)
f = st.gaussian_kde(arr)
f(10000)
import scipy.integrate as si
si.quad(f ,10000 , 15000)[0]
df = pd.read_csv('./pandas/pupils.csv')
df.head()
X = df[['Age','Height','income','rooms','family','type']]
y = df.avg
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
import sklearn.linear_model as sl
model = sl.LinearRegression(normalize=True)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)
y_test
y_pred
sb.distplot(np.abs(y_test - y_pred))
import sklearn.metrics as mt
mt.r2_score(y_test,y_pred)
mt.mean_absolute_error(y_test,y_pred)
np.sqrt(mt.mean_squared_error(y_test,y_pred))
X_test
model.predict([[8,120,30000,4,5,2]])
model.coef_
model.intercept_
8*model.coef_[0] + 120 * model.coef_[1] + 30000 * model.coef_[2] + 4*model.coef_[3] + 5*model.coef_[4] + 2*model.coef_[5] + model.intercept_
import seaborn as sb

import pandas as pd

df = sb.load_dataset('diamonds')
df.head()
df.info()