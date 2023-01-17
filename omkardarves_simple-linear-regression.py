# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

X=data.iloc[:,:-1].values

y=data.iloc[:,1].values
data.head()

y
sns.distplot(data['YearsExperience'],kde=False,bins=10)

plt.show()
sns.pairplot(data)
sns.barplot(x='YearsExperience',y='Salary',data=data)
sns.heatmap(data.corr())
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

y_pred
plt.scatter(X_train,y_train,color='blue')

plt.plot(X_train,lr.predict(X_train),color='red')

plt.title('Salary ~ Experience (Train set)')

plt.xlabel('yrs of exp')

plt.ylabel('salary')

plt.show()
plt.scatter(X_test,y_test,color='blue')

plt.plot(X_train,lr.predict(X_train),color='red')

plt.title('Salary ~ Experience (Test set)')

plt.xlabel('yrs of exp')

plt.ylabel('salary')

plt.show()
from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))

print('MSE:',metrics.mean_squared_error(y_test,y_pred))

print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
