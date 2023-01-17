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
# Question 1

import numpy as np

a=np.array([9,5,4,3,2,6])

b=np.array([5,8,6,9,2,1])

print("CHECK IF B HAS SAME VIEWS TO MEMORY IN A")

print(b.base is a)

print("CHECK IF A HAS SAME VIEWS TO MEMORY IN B")

print(a.base is b)

div_by_3=a%3==0

div1_by_3=b%3==0

print("Divisible By 3")

print(a[div_by_3])

print(b[div1_by_3])

b[::-1].sort()

print("SECOND ARRAY SORTED")

print(b)

print("SUM OF ELEMENTS OF FIRST ARRAY")

print(np.sum(a))
# QUESTION 2

import numpy as np

import pandas as pd

df = pd.read_csv("../input/titanic/train_and_test2.csv")

df.head()



df.dropna(axis=1, how='all')

print(df.head())

print(df.shape)



df[:50].mean()



df[df['Sex']==1].mean()



df['Fare'].max()
# QUESTION 3

import matplotlib.pyplot as plt

Dset = [86,83,86,90,88]

Sub = ['English','Maths','Science','History','Geography']

Color = ['yellow','blue','red','brown','green']

plt.pie(Dset,labels=Sub,colors=Color,startangle=90,shadow=True,explode=[0.0,0.2,0.0,0.0,0.0],autopct='%1.1f%%')

plt.title('Exam Score')

plt.show()
# QUESTION 4import pandas as pd 

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

train = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

X =train.drop("species",axis=1)

y =train["species"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print("F1 Score(macro):",f1_score(y_test, predictions, average='macro'))

print("F1 Score(macro):",f1_score(y_test, predictions, average='macro'))

print("F1 Score(weighted):",f1_score(y_test, predictions, average='weighted'))

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)
