#Q1create two NumPy array by takin user input of data stored in array, check if they have views to same memory, check if elements of arrays are divisible by 3 or not sort 2nd array and find sum of all elements of 1st array
import numpy as np
inp1 = input("Enter first array:")
a = inp1.split()
a = [int(i) for i in a]
inp2 = input("Enter second array:")
b = inp2.split()
b = [int(i) for i in b]
Arr1 = np.array(a)
Arr2 = np.array(b)
print("Array 1 :")
print(Arr1)
print("Array 2 :")
print(Arr2)
print("Do both of these arrays share the same memory :")
print(id(Arr1)==id(Arr2))
div1 = Arr1%3==0
div2 = Arr2%3==0
print("elements of array 1 divisible by 3 are :")
print(Arr1[div1])
print("elements of array 2 divisible by 3 are :")
print(Arr2[div2])
print("Array 2 after sorting is :")
Arr2.sort()
print(Arr2)
print("Sum of elements of array 1 is :")
print (Arr1.sum())
#Q2_Load the titanic dataset,remove mising values from all attributes,find mean valueof first 50 samples, find mean of number of male passengers on the ship, find highest fare paid by any passenger
import pandas as pd
df = pd.read_csv("../input/titanic/train_and_test2.csv")
df.head()

df.dropna(axis=1, how='all')
print(df.head())
print(df.shape)

df[:50].mean()

df[df['Sex']==1].mean()

df['Fare'].max()
#Q3.A student has got some marks. plot a grph to represent the data and highlight the subject with the least marks
import matplotlib.pyplot as plot
marks = [86,83,86,90,88]
subject = ['English','Maths','Science','History','Geography']
colc = ['b','r','b','b','b']
plot.bar(subject,marks,color=colc)
plot.xlabel('SUBJECTS')
plot.ylabel('MARKS')
plot.title('AI ML EXAM Q3')
plot.show()
#Q4 Load the iris dataset, print the confusion matrix and f1_score as computed on the features.
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

print("F1 Score(macro):",f1_score(y_test, predictions,average='macro'))
print("F1 Score(micro):",f1_score(y_test, predictions,average='micro'))
print("F1 Score(weighted):",f1_score(y_test, predictions,average='weighted'))
print("\nConfusion Matrix(below):\n")
confusion_matrix(y_test, predictions)