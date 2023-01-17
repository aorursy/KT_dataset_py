import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/calcofi/bottle.csv')
df.head()
df.columns
df.shape
null_val=df.isnull().sum()

null_val
#the two columns that we need are temperature and salinity

df_final=df[['T_degC','Salnty']]
df_final.head()
sns.scatterplot(data=df_final,x='T_degC',y='Salnty')
final=df_final[:][:500]

final.head()
final=final.dropna(axis=0)
final=final.drop_duplicates()
final.shape
sns.scatterplot(data=final,x='T_degC',y='Salnty')
sns.pairplot(final, kind="scatter")
x=np.array(final['Salnty']).reshape(-1,1)

y=np.array(final['T_degC']).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
reg=LinearRegression()

reg.fit(x,y)
accuracy = reg.score(X_test, y_test)

print(accuracy)
y_pred = reg.predict(X_test)

plt.scatter(X_test, y_test, color='r')

plt.plot(X_test, y_pred, color='g')

plt.show()
reg.intercept_
reg.coef_
rmse = np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))

print(rmse)