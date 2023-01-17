#1. Create two NumPy array(unsorted) by taking user input of all the data stored in the array,
# check if they have views to the same memory and print the result,
# check if any of the elements in the two arrays are divisible by 3 or not and print the output,
# sort the 2nd array and print it, find sum of all elements of array 1.

import numpy 

a = []
b = []
a_size = int(input("Enter size of array A"))
for i in range(a_size):
    a.append(int(input("Enter element: ")))
A = numpy.array(a)
    
b_size = int(input("Enter size of array B"))
for i in range(b_size):
    b.append(int(input("Enter element: ")))
B = numpy.array(b)

print("Checking if A and B refer to the same memory")
print(id(A)==id(B))
print("Checking if any of the elements in the two arrays are divisible by 3 or not")
flag = bool()
for i in a:
    if(i%3==0):
        flag=True;
        break;
for i in b:
    if(i%3==0):
        flag=True;
        break;
print(flag)
print("The data after checking is")
print(A%3==0)
print(B%3==0)
print("\nSorting and printing B")
print(numpy.sort(B))
print("\nSum of elements of A",numpy.sum(A))
# 2. Load the titanic dataset, remove missing values from all attributes,
# find mean value of first 50 samples, find the mean of the number of male passengers( Sex=1)
# on the ship, find the highest fare paid by any passenger. 

import numpy as np
import pandas as pd

data = pd.read_csv("../input/titanic/train_data.csv")
data.dropna(inplace=True)
print("\nThe dataset after removal is\n",data.describe())


#print("Mean of age of first 50 passengers")
#mean_age = data['Age'].head(50)
#mn = mean_age.mean()
#print(mn*100)

print("Mean of first 50 male passengers")
males = data["Sex"].head(50)
print(males.mean())



fare_max = data['Fare']

max = fare_max.max()

print("Highest fair paid by passenger",max)
# 3. A student has got the following marks ( English = 86, Maths = 83, Science = 86, History =90, Geography = 88).
# Wisely choose a graph to represent this data such that it justifies the purpose of data visualization.
# Highlight the subject in which the student has got least marks. 

import matplotlib.pyplot as plt
subj=['English', 'Maths', 'Science', 'History','Geography']
marks=[86,83,86,90,88]
cols=['c','m','r','b','g']
plt.bar(subj,marks,color=['green','red','green','green','green'])
plt.title('SUBJECT VS MARKS')
plt.xlabel("SUBJECT")
plt.ylabel("MARKS")
# 4. Load the iris dataset, print the confusion matrix and f1_score as computed on the features.

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