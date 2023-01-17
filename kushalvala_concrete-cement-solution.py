import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
#Getting Data Set 
df = pd.read_csv('../input/Concrete.csv')
df.head()


# Feature variable "csMPa" is the target variable.
df.describe()
df.info()

#Since there are no missing values: No Imputations required
#Explotary Data Analysis

print(df['cement'].mean())
plt.hist(df.cement, bins=8)
plt.figure(figsize=(10,10))
df_corr = df.corr()
sns.heatmap(df_corr, annot=True)
df.corr()
plt.scatter(df['cement'],df['csMPa'])
df.head()
X = df.iloc[:,0:-1]
print(X.head())
y = df.iloc[:,-1]
y.head()
#Splitting Test and Train Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 101)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)
#Using Multi-Linear Regression 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
#Using Validation Parameters
from sklearn.metrics import r2_score, mean_squared_error

print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))