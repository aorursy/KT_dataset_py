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
import pandas as pd
data = pd.read_csv("../input/titanic/train_and_test2.csv")
new_dataset = data.dropna(axis = 0, how ='any') 
new_dataset.describe()

x = new_dataset.head(50)
print("The mean of First 50 samples :", x.mean())
print("The mean of first 50 male passengers")
male = data["Sex"].head(50)
print(male.mean())
max_fare=data['Fare']
max = max_fare.max()
print("Highest fare paid by a passenger",max)
import pandas as pd
data= pd.read_csv("../input/titanic/train_and_test2.csv")
miss=data.dropna(axis=1,how='all')
print(miss)
data1=data.head(50)
print(data.mean())
male=data['Sex']==1
print(male.mean())
print(data['Fare'].max())
import matplotlib.pyplot as plt
Subjects = ['English','Maths','Science','History','Geography']
Marks =[86,83,86,90,88]
      
plt.bar(Subjects,Marks,color=['GREEN','red','GREEN','GREEN','GREEN'])
plt.title('Marks Graph')
plt.xlabel('SUBJECTS')
plt.ylabel('MARKS')
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