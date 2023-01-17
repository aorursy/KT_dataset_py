arr1 = []
a = int(input("Size of array:"))
for i in range(a):
    arr1.append(float(input("Element:")))
arr1 = np.array(arr1)
arr2 = []
a = int(input("Size of array:"))
for i in range(a):
    arr2.append(float(input("Element:")))
arr2 = np.array(arr2)

print('----------------------------')

# Checking if arr1 has views to arr2 memmory
print(arr1.base is arr2)
# Checking if arr2 has views to arr1 memmory
print(arr2.base is arr1)

print('----------------------------')

div1=arr1%3==0
print("Divisible by 3 in arr1")
print(arr1[div1])

print('----------------------------')

div2=arr2%3==0
print("Divisible by 3 in arr2")
print(arr2[div2])

print('----------------------------')

print('sorted arr2 is:')
print(np.sort(arr2))

print('----------------------------')

sum = np.sum(arr1)
print(f"The sum of all elements in arr1 is {sum}")
titanic_data=pd.read_csv("../input/titanic/train_and_test2.csv")
print("TITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)

titanic_data.dropna(axis=1, how='all')
print("------\nTITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)

print("------\nMean value of first 50 samples: \n",titanic_data[:50].mean())

print("------\nMean of the number of male passengers( Sex=1) on the ship :\n",titanic_data[titanic_data['Sex']==1].mean())

print("------\nHighest fare paid by any passenger: ",titanic_data['Fare'].max())
from matplotlib import pyplot as plt
SUBJECTS=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
MARKS=[86,83,86,90,88] 
tick_label=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.3,color=['blue','red','blue','blue','blue'])
plt.xlabel('SUBJECTS') 
plt.ylabel('MARKS')
plt.title("STUDENT's MARKS DATASET")
plt.show()
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