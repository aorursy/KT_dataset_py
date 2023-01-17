# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
data=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
# Any results you write to the current directory are saved as output.
data.info()
data=data.drop('Serial No.',axis=1)
data.head()
import seaborn as sns
sns.pairplot(data=data, kind='reg')
data.columns
y=data['Chance of Admit ']
x=data.drop('Chance of Admit ',axis=1)
x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(x_train,y_train)
model.predict(x_test)
predict=model.predict(x_test)
y_test.head()
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,predict)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predict)
marks=['320','120','4','4.0','3.0','8.00','1']
marks=pd.DataFrame(marks).T
marks
model.predict(marks)
value=model.predict(marks)
print("My chance of selection:", value)