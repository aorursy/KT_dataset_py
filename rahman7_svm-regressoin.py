# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset:
df=pd.read_csv("../input/Position_Salaries.csv")
df
# data Visualization:
sns.pairplot(df)
sns.distplot(df['Salary'],bins=20,kde=False,)
sns.scatterplot(df['Level'],df['Salary'],data=df)
sns.countplot(df['Salary'])
# make a dataset in the form of depenedent and independent:
X=df.iloc[:,1:2].values
y=df.iloc[:,2].values
y=y.reshape(len(y),1)
X
y
# spliting the datset into train and test set:
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#feature Scalling:
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

# fitting the dataset on SVR model:
from sklearn.svm import SVR
regressor=SVR()
regressor.fit(X,y)
# prediction of new result:
y_pred=regressor.predict(X)
y_pred
# visualization of SVR model:
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salary Vs level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# visualization of SVR model(higher magifiaction result):
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Salary Vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
