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
#Create two NumPy array(unsorted) by taking user input of all the data stored in the array, check if they have views to the same memory and print the result, check if any of the elements in the two arrays are divisible by 3 or not and print the output, sort the 2nd array and print it, find sum of all elements of array 1

#Question 1

import numpy as np

a=np.array([2,7,4,3,2,])

b=np.array([9,8,3,9,2,])

print("CHECKING B HAS SAME VIEWS TO MEMORY IN A")

print(b.base is a)

print("CHECKING A HAS SAME VIEWS TO MEMORY IN B")

print(a.base is b)

div_by_3=a%3==0

div1_by_3=b%3==0

print("Divisible By 3")

print(a[div_by_3])

print(b[div1_by_3])

b[::-1].sort()

print("SECOND ARRAY")

print(b)

print("SUM OF ELEMENTS OF FIRST ARRAY")

print(np.sum(a))
#Load the titanic dataset, remove missing values from all attributes, find mean value of first 50 samples, find the mean of the number of male passengers( Sex=1) on the ship, find the highest fare paid by any passenger. 

#QUESTION 2

import pandas as pd

df = pd.read_csv("../input/titanic/train_and_test2.csv")

df.head()

df.dropna(axis=1, how='all')

print(df.head())

print(df.shape)

df[:50].mean()

df[df['Sex']==1].mean()

df['Fare'].max()
#A student has got the following marks ( English = 86, Maths = 83, Science = 86, History =90, Geography = 88). Wisely choose a graph to represent this data such that it justifies the purpose of data visualization. Highlight the subject in which the student has got least marks. 

#QUESTION 3

from matplotlib import pyplot as plt

SUBJECTS=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]

MARKS=[86,83,86,90,88] 

tick_label=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]

plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.4,color=['green','red','yellow','blue','orange'])

plt.xlabel('SUBJECT') 

plt.ylabel('MARKS')

plt.title("STUDENT's MARKS DATASET")

plt.show()
#Load the iris dataset, print the confusion matrix and f1_score as computed on the features

#QUESTION 4

import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix



# import some data to play with

iris = datasets.load_iris()

X = iris.data

y = iris.target

class_names = iris.target_names



# Split the data into a training set and a test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



# Run classifier, using a model that is too regularized (C too low) to see

# the impact on the results

classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)



np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(classifier, X_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize=normalize)

    disp.ax_.set_title(title)



    print(title)

    print(disp.confusion_matrix)



plt.show()