# A student has got the following marks ( English = 86, Maths = 83, Science = 86, History =90, Geography = 88). 
# Wisely choose a graph to represent this data such that it justifies the purpose of data visualization. 
# Highlight the subject in which the student has got least marks. 

import matplotlib.pyplot as plt
sub=["English","Maths","Science","Hsitory","Geography"]
marks=[86,83,86,90,88]
cols=["y","c","m","gold","orange"]
plt.pie(marks,labels=sub,colors=cols,startangle=90,shadow=True,explode=(0,0.3,0,0,0),autopct='%1.2f%%')
plt.title("Pie Graph for Data Visualization")
plt.show()

# Load the iris dataset, print the confusion matrix and f1_score as computed on the features.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('../input/iris/Iris.csv', error_bad_lines=False)
df = df.drop(['Id'], axis=1)
df['Species'] = pd.factorize(df["Species"])[0] 
Target = 'Species'
Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

model = LogisticRegression(solver='lbfgs', multi_class='auto')
Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]
x2 = x[Target]
y1 = y[Features]
y2 = y[Target]

nb_model = GaussianNB() 
nb_model.fit(X=x1, y=x2)
result= nb_model.predict(y[Features]) 

f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)

# Load the titanic dataset, remove missing values from all attributes, find mean value of first 50 samples, 
# find the mean of the number of male passengers( Sex=1) on the ship, find the highest fare paid by any passenger.

titanic_data=pd.read_csv("../input/titanic/train_and_test2.csv")
print("TITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

titanic_data.dropna(axis=1, how='all')
print("_______________________________________________________\nTITANIC DATASET : ")
print(titanic_data.head())
print("TITANIC DATASET SHAPE : ",titanic_data.shape)
print(titanic_data.shape)

print("________________________________________________________\nMean value of first 50 samples: \n",titanic_data[:50].mean())

print("_______________________________________________________\nMean of the number of male passengers( Sex=1) on the ship :\n",titanic_data[titanic_data['Sex']==1].mean())

print("_______________________________________________________\nHighest fare paid by any passenger: ",titanic_data['Fare'].max())

# Creating two NumPy array by takin user input of data stored in array, check if they have views to same memory, 
# check if elements of arrays are divisible by 3 or not sort 2nd array and find sum of all elements of 1st array
import numpy as np
inp1 = input("Enter first array:")
a = list(map(int, inp1.rstrip().split()))
inp2 = input("Enter second array:")
b = list(map(int, inp2.rstrip().split()))
Arr1 = np.array(a)
Arr2 = np.array(b)
print("Array 1 :",Arr1)
print("Array 2 :",Arr2)

print("Do both of these arrays share the same memory : ",id(Arr1)==id(Arr2))

print("Any Elements of array 1 that are divisible by 3 : ", Arr1%3==0)
print("Elements of array 1 divisible by 3 are : ",Arr1[Arr1%3==0])
print("Any Elements of array 2 that are divisible by 3 : ", Arr2%3==0)
print("Elements of array 2 divisible by 3 are : ",Arr2[Arr2%3==0])

Arr2.sort()
print("Array 2 after sorting is :")
print(Arr2)
print("Sum of elements of array 1 is :")
print (Arr1.sum())