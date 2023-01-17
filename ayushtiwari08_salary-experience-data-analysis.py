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
import pandas as pd

data = pd.read_csv("../input/salary/Salary.csv")

data.head()

data.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

px.line(data,x = "YearsExperience" , y = "Salary",title="Years of Experiance Vs Salary")

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set_palette("BrBG")

fig=sns.regplot(x=data["YearsExperience"],y=data["Salary"])

fig
px.scatter(data,x = "YearsExperience" ,  y= "Salary",color="Salary",title="Distribution of Data for Salaries vs Years of experience")
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

#Selecting Corresponding Features

X = data['YearsExperience'].values

y = data['Salary'].values



X = X.reshape(-1,1)

y = y.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=100)

plt.scatter(x_train,y_train,color='blue')

plt.xlabel('Years of experience')

plt.ylabel('Salary in Dollars')

plt.title('Training data')

plt.show()
lr = LinearRegression()

lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

print(f"Train accuracy {round(lr.score(x_train,y_train)*100,2)} %")

print(f"Test accuracy {round(lr.score(x_test,y_test)*100,2)} %")
plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,y_predict)

plt.xlabel("Years of Experience")

plt.ylabel("Salary in Dollors")

plt.title("Trained model plot")

plt.plot
import numpy as np

sampledata = np.array([15,1.5,7.3,9.65])

sampledata = sampledata.reshape(-1,1)

sample_salary = lr.predict(sampledata)

for salary in sample_salary:

    print(f"$ {salary}")