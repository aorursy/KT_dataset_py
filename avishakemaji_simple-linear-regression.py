# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
data.head(5)
data.isnull().sum()
data.info()
plt.figure(figsize=(25,25))
sns.heatmap(data.isnull())# This shows there is no null value
sns.distplot(data['YearsExperience'])
sns.heatmap(data.corr(),annot=True)
data.describe().T
x=data.iloc[:,0]
y=data['Salary']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=0)
plt.figure(figsize=(10,10))
plt.scatter(x_train,y_train,c='r',marker='o',s=50)
plt.title('Years of Experience vs Salary Relationship',fontsize=30)
plt.xlabel('Years of Experience',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.show()
import seaborn as sns
plt.figure(figsize=(16,9))
ax=sns.barplot(x_train,y_train)
ax.set(title='Years of Experience vs Salary')


x_mean=np.mean(x_train)
y_mean=np.mean(y_train)
print(x_mean,'',y_mean)
n=len(x_train)
s_yx=np.sum(x_train*y_train)-n*x_mean*y_mean
s_xx=np.sum(x_train*x_train)-n*(x_mean)**2
print(s_yx,'',s_xx)
b1=s_yx/s_xx
bo=y_mean-b1*x_mean
print("Coeffients: ",b1)
print("Intercept: ",bo)
y_pred=bo+b1*x_train

plt.figure(figsize=(10,10))
plt.scatter(x_train,y_train,c='r',marker='o',s=50)
plt.plot(x_train,y_pred)
plt.title('Years of Experience vs Salary Relationship',fontsize=30)
plt.xlabel('Years of Experience',fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.show()
y_test_pred=bo+b1*x_test
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("MSE= " ,mean_squared_error(y_test,y_test_pred))
print("R2_score: ",r2_score(y_test,y_test_pred))
plt.title('Residual Plot',size=16)
sns.residplot(y_test,y_test_pred,color='r')
plt.xlabel('Y_pred',size=12)
plt.ylabel('Residues',size=12)