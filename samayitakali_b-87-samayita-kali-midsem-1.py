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
#Question 1

#Create two NumPy array(unsorted) by taking user input of all the data stored in the array, 

#check if they have views to the same memory and print the result, 

#check if any of the elements 

#in the two arrays are divisible by 3 or not and print the output, 

#sort the 2nd array and print it, find sum of all elements of array 1.

import numpy as np

x= np.array([2,1,9,6,4])

y= np.array([1,4,2,3,9])



print("check if they have views to the same memory:")

print("x is y: ",x is y)

print("x location:",id(x),"\ty location:",id(y))

print("x:",x)

print("y:",y)



div_by_3_1 = x%3==0

div_by_3_2 = y%3==0

print("\n")

print("Check divisibility by 3")

print("\tfor array x")

print(div_by_3_1)

print(x[div_by_3_1])

print("\n\tfor array y")

print(div_by_3_2)

print(y[div_by_3_2])



print("\nSum of all elements of array x: ",x.sum())



print("\nSorted Array y: ",np.sort(y))
#Question 2

#Load the titanic dataset, remove missing values from all attributes, 

#find mean value of first 50 samples, 

#find the mean of the number of male passengers( Sex=1) on the ship, 

#find the highest fare paid by any passenger. 



import pandas as pd

data= pd.read_csv("../input/titanic/train_and_test2.csv")

print("Remove missing values from all attributes:\n")

miss=data.dropna(axis=1,how='all')

print(miss)



print("\n\nMean value of first 50 samples:\n ")

data1=data.head(50)

print(data.mean())



male=data['Sex']==1

print("\nMean of Male passengers on Ship: ",male.mean())



print("\nHighest fare paid by any passenger: ",data['Fare'].max())



#Question 3

# A student has got the following marks ( English = 86, Maths = 83, Science = 86, History =90, Geography = 88).

# Wisely choose a graph to represent this data such that it justifies the purpose of data visualization. 

# Highlight the subject in which the student has got least marks. 



import matplotlib.pyplot as plt

Subjects = ['English','Maths','Science','History','Geography']

Marks =[86,83,86,90,88]

 

#BAR GRAPH

plt.bar(Subjects,Marks,color=['GREEN','red','GREEN','GREEN','GREEN'])

plt.title('Marks Graph')

plt.xlabel('SUBJECTS')

plt.ylabel('MARKS')
#Question 4

#Load the iris dataset, print the confusion matrix and f1_score as computed on the features.



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



a = pd.read_csv("../input/iris/Iris.csv", error_bad_lines=False)

a = a.drop(['Id'], axis=1)

a['Species'] = pd.factorize(a["Species"])[0] 

Target = 'Species'

Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



model = LogisticRegression(solver='lbfgs', multi_class='auto')

Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



x, y = train_test_split(a, 

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