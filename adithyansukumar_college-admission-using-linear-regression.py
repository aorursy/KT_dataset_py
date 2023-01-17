import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv',sep=r'\s*,\s*')
df.head()
df.info()

df.describe()
sns.scatterplot(x='Chance of Admit',y='GRE Score',data=df)
sns.scatterplot(x='Chance of Admit',y='TOEFL Score',data=df)
plt.hist(df['GRE Score'])
plt.xlabel('GRE SCORE')
plt.hist(df['TOEFL Score'])
plt.xlabel('TOEFL SCORE')
plt.scatter(df['University Rating'],df['GRE Score'],)
sns.countplot(df['Research'])
sns.scatterplot(y='Chance of Admit',x='Research',data=df)
sns.jointplot(x='CGPA',y='Chance of Admit',data=df)
sns.heatmap(df.corr())
sns.pairplot(df)
sns.jointplot(x='GRE Score',y='TOEFL Score',data=df)
df.drop('Serial No.',axis=1,inplace=True)
df.head()
x=df.drop('Chance of Admit',axis=1)
y=df['Chance of Admit']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
predictions=lr.predict(x_test)
print("intercept=",lr.intercept_)
print("coefficient=",lr.coef_)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.scatter(y_test,predictions)
