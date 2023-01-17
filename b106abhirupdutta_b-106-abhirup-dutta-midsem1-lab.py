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
#Q1. Create two NumPy array(unsorted) by taking user input of all the data stored in the array, check if they have views to the same memory 
#and print the result, check if any of the elements in the two arrays are divisible by 3 or not and print the output, sort the 2nd array and print it, 
#find sum of all elements of array 1.
import numpy as np
arr1 = []
a = int(input("Size of array:"))
for i in range(a):
    arr1.append(int(input("Element:")))
arr1 = np.array(arr1)
arr2 = []
a = int(input("Size of array:"))
for i in range(a):
    arr2.append(int(input("Element:")))
arr2 = np.array(arr2)
# Checking if arr1 has views to arr2 memmory
print(arr1.base is arr2)
# Checking if arr2 has views to arr1 memmory
print(arr2.base is arr1)
div1=arr1%3==0
print("Divisible by 3")
print(arr1[div1])
div2=arr2%3==0
print("Divisible by 3")
print(arr2[div2])
print(np.sort(arr2))
sum = np.sum(arr1)
print(f"The sum of all elements in arr1 is {sum}")
#Q2. Load the titanic dataset, remove missing values from all attributes, find mean value of first 50 samples, 
#find the mean of the number of male passengers( Sex=1) on the ship, find the highest fare paid by any passenger. 
titanic_data=pd.read_csv("../input/titanic/train_and_test2.csv")
print("TITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

titanic_data.dropna(axis=1, how='all')
print("__\nTITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

print("__\nMean value of first 50 samples: \n",titanic_data[:50].mean())

print("__\nMean of the number of male passengers( Sex=1) on the ship :\n",titanic_data[titanic_data['Sex']==1].mean())

print("__\nHighest fare paid by any passenger: ",titanic_data['Fare'].max())
#Q3. A student has got the following marks ( English = 86, Maths = 83, Science = 86, History =90, Geography = 88). 
#Wisely choose a graph to represent this data such that it justifies the purpose of data visualization. 
#Highlight the subject in which the student has got least marks. 
from matplotlib import pyplot as plt
SUBJECTS=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
MARKS=[86,83,86,90,88] 
tick_label=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]  
plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.8,color=['green','red','green','green','green'])   
plt.xlabel('SUBJECTS') 
plt.ylabel('MARKS')  
plt.title("STUDENT's MARKS DATASET")
plt.show()
#Q4. Load the iris dataset, print the confusion matrix and f1_score as computed on the features.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


X = train.drop("species",axis=1)
y = train["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print("F1 Score(macro):",f1_score(y_test, predictions, average='macro'))
print("F1 Score(micro):",f1_score(y_test, predictions, average='micro'))
print("F1 Score(weighted):",f1_score(y_test, predictions, average='weighted')) 
print("\nConfusion Matrix(below):\n")
confusion_matrix(y_test, predictions)