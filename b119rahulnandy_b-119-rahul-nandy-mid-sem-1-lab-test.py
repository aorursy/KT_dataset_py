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

#1. Load a dataset of your choice, display the first 3 rows, display a row of that dataset having missing values and replace missing values with Nan

import pandas as pd
import numpy as np


exam_data  = {'name': ['Natasa', 'Eva', 'Ritesh', 'James', 'John', 'Michael', 'Alisa', 'Laura', 'Kevin', 'Roma'],
        'score': [10.5, 8.5, 14, np.nan, 9, 5, 7, np.nan, 8, 19],
        'attempts': [1, 3, 2, 0, 0, 3, 5, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)

df = pd.DataFrame(exam_data , index=labels)
print("--First three rows of the data frame:\n")
print(df.iloc[:3])

df = pd.DataFrame(exam_data , index=labels)
print("\n--Rows where score is missing:\n")
print(df[df['score'].isnull()])
#2. Given score of CSK, KKR, DC and MI such that no two team has same score, chalk out an appropriate graph for best display of the scores.
#Also highlight the team having highest score in the graph

ipl_data  = {'name': ['CSK', 'MI', 'KKR', 'DC','RPG','RAJROYAL','SRH'],
        'score': [12.5, 9, 16.5,  9, 20, 14.5,  8],
        'attempts': [1, 3, 2, 3, 2, 3, 1],
        }

df = pd.DataFrame(ipl_data )
df.head()



from pandas import DataFrame
col=['c' for i in range(8)]
col[4]='red'
df.plot(x ='name', y='score', kind = 'bar',color=col)

# 3.Take two numpy array of your choice, find the common items between the arrays and remove the matching items but only from one array such that they exist in the second one.

import numpy as np
a = np.array([11,23,17,34,5])
print("Array 1: ",a)
b = np.array([11,34,27])
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


#4.Write a program to display the confusion matrix and f1_score on the titanic dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train = pd.read_csv("../input/titanic/train_data.csv")
                    


X = train.drop("Survived",axis=1)
y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print("F1 Score:",f1_score(y_test, predictions))
 
print("\nConfusion Matrix(below):\n")
confusion_matrix(y_test, predictions)