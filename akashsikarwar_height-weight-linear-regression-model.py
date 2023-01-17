# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/heights-and-weights/data.csv")
df.head()
X=df.iloc[:,:-1].values

y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train), color='blue')

plt.title('Training Set Result')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
plt.scatter(X_test,y_test,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title('Test Set Result')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
from sklearn.metrics import r2_score,mean_squared_error

coefficient_of_dermination = r2_score(y_pred, y_test)

print('R2 Score = ',coefficient_of_dermination)

error=mean_squared_error(y_pred, y_test)

print('Error = ',error)
plt.scatter(X_test,y_test,color='red')

plt.plot(X_test,y_pred,color='blue')

plt.title('Test Set Result')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()