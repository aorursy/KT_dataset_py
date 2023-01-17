import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/USA_Housing.csv')
df.head()
df.info()
df.info()
df.columns
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(),annot=True)
df.columns
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
print(lm.coef_)
pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
pred = lm.predict(X_test)
pred
from matplotlib import pyplot as plt

import numpy as np

import matplotlib

plt.scatter(y_test,pred)