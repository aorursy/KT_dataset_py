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
# 1.Load a dataset of your choice, display the first 3 rows, display a row of that dataset having missing values and replace missing values with Nan.

import pandas as pd

import numpy as np

record_data = pd.read_csv("../input/records/university_records.csv")

df=pd.DataFrame(record_data)

print(record_data)

print('\n\n')

print(df.head(3))

print('\n\n')

null=df[df['CGPA'].isnull()]

print(null)

print('\n\n')















#2.Given score of CSK, KKR, DC and MI such that no two team has same score, chalk out an appropriate graph for best display of the scores. 

#Also highlight the team having highest score in the graph. 

from matplotlib import pyplot as plt



team=['CSK','KKR','DC','MI']

points=[3,7,9,5]



plt.bar(team, points, width = 0.8, color=['red','green','green','green'])



plt.xlabel('Team') 

plt.ylabel('Points')

plt.title('Team Points Chart')

plt.show()
#3.Take two numpy array of your choice, find the common items between the arrays 

#and remove the matching items but only from one array such that they exist in the second one.

import numpy as np

import pandas as pd



a = np.array([1,3,5,9,11,13])



b = np.array([9,11,13])



c = np.intersect1d(a,b) #printing the common items

print(c)

print("\n")

for i in b:

    for j in a:

        if i == j:

            a = a[a!=j] #removing the common items from the array "a"

print(a)

print("\n")

#4. Write a program to display the confusion matrix and f1_score on the titanic dataset.

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



train = pd.read_csv("../input/titanic/train_data.csv")





X = train.drop("Survived",axis=1)

y = train["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print("F1 Score:",f1_score(y_test, predictions))

 

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)