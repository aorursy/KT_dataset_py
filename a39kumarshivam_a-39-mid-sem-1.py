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
# 1. Create two NumPy array(unsorted) by taking user input of all the data stored in the array, 
# check if they have views to the same memory and print the result, check if any of the elements 
#in the two arrays are divisible by 3 or not and print the output, sort the 2nd array and print it, 
#find sum of all elements of array 1.
import numpy as np
l = []
m = []
n1 = int(input("Size of array1 :"))

for i in range(n1):
    i=float(input("Enter the elements:"))
    l.append(i)
arr1 = np.array(l)
print(arr1)
n2 = int(input("Size of array2 :"))
for j in range(n2):
    j=float(input("Enter the elements:"))
    m.append(j)
arr2 = np.array(m)
print(arr2)
print("Check if arr2 has same views to memory in arr1 :")
print(arr2.base is arr1)
print("Check if arr1 has same views to memory in arr2 :")
print(arr1.base is arr2)
x = arr1%3 ==0
print(arr1[x])
y = arr2%3 ==0
print(arr2[y])
arr2[::-1].sort()
print("Second array after sorting is:")
print(arr2)
print("Sum of elements of first array is : ")
print(np.sum(arr1))


    


# 2. Load the titanic dataset, remove missing values from all attributes, find mean value of first 50 samples, 
# find the mean of the number of male passengers( Sex=1) on the ship, find the highest fare paid by 
# any passenger.
import pandas as pd
df = pd.read_csv("../input/titanic/train_and_test2.csv")
df.head()

df.dropna(how='all')
print(df.head())
print(df.shape)

df[df['Sex']==1].mean()
df[:50].mean()
df['Fare'].max()



# 3. A student has got the following marks ( English = 86, Maths = 83, Science = 86, History =90, Geography = 88).
# Wisely choose a graph to represent this data such that it justifies the purpose of data visualization. 
# Highlight the subject in which the student has got least marks.
import matplotlib.pyplot as plt
import numpy as np
Subject =["English","Math","Science","History","Geography"]
Marks =[86,83,86,90,88]
plt.bar(Subject,Marks)
plt.bar(Subject[1],Marks[1],color='g')
plt.xlabel('Subjects')
plt.ylabel("Marks")
plt.title('Marks of the student')
plt.show()
# 4.Load the iris dataset, print the confusion matrix and f1_score as computed on the features
import pandas as pd
