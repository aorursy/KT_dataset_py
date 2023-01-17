import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation. plotting graphs from datasets
#------------------------------------------------------------------------------------------------------------------

#---Question 1-----------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------

arr1 = np.array([])

arr2 = np.array([])

t=0

s=int(input("Enter size of array 1: "))

print("Enter elements of array 1: ")

for i in range(s):

    t=int(input())

    arr1 = np.append(arr1, [t])



s=int(input("Enter size of array 2: "))

print("Enter elements of array 2: ")

for i in range(s):

    t=int(input())

    arr2 = np.append(arr2, [t])
print(arr1 is arr2)

print(id(arr1), end=', ')

print(id(arr2))
mult=[]

flag=False

for i in arr1:

    if i%3==0:

        flag=True

        mult.append(['arr1',i])

for i in arr2:

    if i%3==0:

        flag=True

        mult.append(['arr2',i])

print(flag)

print(mult)
arr2= np.sort(arr2)

arr2
s = np.sum(arr1)

print(s)
#------------------------------------------------------------------------------------------------------------------

#---Question 2-----------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("../input/titanic/train_and_test2.csv")

df.head()
df.dropna(axis=1, how='all')

print(df.head())

print(df.shape)
df[:50].mean()
df[df['Sex']==1].mean()
df['Fare'].max()
#------------------------------------------------------------------------------------------------------------------

#---Question 3-----------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------

Dset = [86,83,86,90,88]

Sub = ['English','Maths','Science','History','Geography']

Color = ['yellow','blue','red','brown','green']

plt.pie(Dset,labels=Sub,colors=Color,startangle=90,shadow=True,explode=[0.0,0.2,0.0,0.0,0.0],autopct='%1.1f%%')

plt.title('Exam Score')

plt.show()
#------------------------------------------------------------------------------------------------------------------

#---Question 4-----------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression



df = pd.read_csv("../input/iris/Iris.csv")

df.head()
## X = df.drop("Species", axis=1)

y = df["Species"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



estimate = y_test

prediction= logmodel.predict(X_test)



print("F1 Score: ", f1_score(estimate,prediction,average="weighted"))



print("\nConfusion Matrix shown below:\n")

confusion_matrix(estimate,prediction)