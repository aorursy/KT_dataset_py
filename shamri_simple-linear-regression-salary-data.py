# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
%matplotlib inline
#importing the csv file 
data = pd.read_csv("../input/Salary.csv")
#to see the head and describe the statistics
data.head()
data.describe()
x=data['YearsExperience']
y=data['Salary']
# drawing a line plot
sns.lineplot(x,y)
#to draw regplot
sns.regplot(x,y)

#to draw scatterplot using seaborn package
sns.scatterplot(x,y,color="red")

#Another method of scatter diagram using matplotlib
plt.scatter(x,y,color="green")
plt.xlabel('years of Experience')
plt.ylabel('salary in $')
plt.show()
x=data[['YearsExperience']]
y=data[['Salary']]
#divide data into train and split 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=100)
#Building linear regression model or training the model
lm=LinearRegression()
lm.fit(x_train,y_train)

#Predicting y values for the test dataset based on trained model
y_predict=lm.predict(x_test)

y_predict

#Finding the train model accuracy
round(lm.score(x_train,y_train)*100,2)
#Finding test model accuracy
round(lm.score(x_test,y_test)*100,2)