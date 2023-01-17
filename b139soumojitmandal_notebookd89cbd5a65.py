# 1. Load a dataset of your choice, display the first 3 rows, display a row of that dataset having missing values and replace missing values with Nan.

import pandas as pd 

employee_info = { 'Name' :['rakesh','mamata','shewli','tommy','kronten'],

                 'salary':['90000','65000','50000','78000','NaN'],

                 'age':['23','24','22','NaN','34']}

labels = ['1','2','3','4','5']

df=pd.DataFrame(employee_info,index=labels)

print (df[0:3])

import pandas as pd







# Read a dataset with missing values

flights = pd.read_csv("../input/titanic/train_and_test2.csv")

  # Select the rows that have at least one missing value

flights[flights.isnull().any(axis=1)].head()
#2. Given score of CSK, KKR, DC and MI such that no two team has same score, chalk out an appropriate graph for best display of the scores. Also highlight the team having highest score in the graph. 

import matplotlib.pyplot as plt  

teams = ['CSK', 'KKR', 'DC', 'MI'] 

slices = [99, 79, 88, 62] 

colors = ['r', 'y', 'g', 'b']

plt.pie(slices, labels = teams, colors=colors,  

        startangle=90, shadow = True, explode = (0.5, 0, 0, 0), 

        radius = 1.2, autopct = '%1.1f%%')  

plt.show()
#3. Take two numpy array of your choice, find the common items between the arrays and remove the matching items but only from one array such that they exist in the second one.

import numpy as np

a = np.array([1,2,3,4,5])

print("Array 1: ",a)

b = np.array([1,4,7])

print("Array 2: ",b)

print("Common values between two arrays:")

print(np.intersect1d(a, b))



for i, val in enumerate(a):

    if val in b:

        a = np.delete(a, np.where(a == val)[0][0])

for i, val in enumerate(b):

    if val in a:

        a = np.delete(a, np.where(a == val)[0][0])

print("Arrays after deletion of common elements : ")

print(a)

print(b)
#4. Write a program to display the confusion matrix and f1_score on the titanic dataset.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('../input/titanic/train_and_test2.csv')

df

x=df[['Passengerid','Age','Fare','Sex']]

x
y=df[['2urvived']]

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.28)

from sklearn.linear_model import LogisticRegression

logisticr=LogisticRegression()

logisticr.fit(x_train,y_train)

prediction=logisticr.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,prediction))

print(confusion_matrix(y_test,prediction))