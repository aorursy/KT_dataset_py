import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

df = df.drop(['Serial No.'],axis=1)

df.head()
df.columns
df.info()

df.isna().any()
df.describe()
pg = sns.PairGrid(df)

pg.map_diag(sns.distplot)

pg.map_upper(plt.scatter,alpha=0.5)

pg.map_lower(sns.kdeplot)
sns.heatmap(df.corr(), annot=True, linewidths=.5,cmap='mako')
sns.distplot(df['Chance of Admit '])
x = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']]

y = df['Chance of Admit ']



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression

linr = LinearRegression()

linr.fit(x_train, y_train)
coef = pd.DataFrame(linr.coef_,x.columns,columns=['Coefficients'])

coef
pred = linr.predict(x_test)

sns.regplot(x=y_test, y=pred)
from sklearn import metrics
metrics.mean_absolute_error(y_test,pred)
np.sqrt(metrics.mean_squared_error(y_test,pred))