import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
data_frame = pd.read_csv(r"../input/Salary_Data.csv")
data_frame.shape
data_frame.head()
data_frame.describe()
sns.distplot(data_frame['YearsExperience'],kde=False,bins=10)
sns.countplot(y='YearsExperience',data=data_frame)
sns.heatmap(data_frame.corr(),annot=True)
data_frame.dtypes
data_frame.isnull().sum()
#Splitting the Dataset

X=data_frame.iloc[:,:-1].values

y=data_frame.iloc[:,1].values

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)
from sklearn.linear_model import LinearRegression

lr_reg = LinearRegression()
lr_reg.fit(X_train,y_train)
y_pred = lr_reg.predict(x_test)
plt.scatter(X_train,y_train,color="blue")

plt.plot(X_train,lr_reg.predict(X_train),color = 'green')

plt.title("Salary-Experience Train data visualization")

plt.xlabel("Year of Experience")

plt.ylabel("salary")

plt.show()
plt.scatter(x_test,y_test,color="blue")

plt.plot(X_train,lr_reg.predict(X_train),color = 'green')

plt.title("Salary-Experience Test data visualization")

plt.xlabel("Year of Experience")

plt.ylabel("salary")

plt.show()
from sklearn import metrics

import numpy as np

print('Mean Abs Error MAE      :' ,metrics.mean_absolute_error(y_test,y_pred))

print('Mean Sqrt Error MSE     :' ,metrics.mean_squared_error(y_test,y_pred))

print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('r2 value                :' ,metrics.r2_score(y_test,y_pred))