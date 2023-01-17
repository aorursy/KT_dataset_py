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
import pandas as pd

import numpy as np

data=pd.read_csv("../input/flight-route-database/routes.csv")

print(data.head(11))

print(data[data.isnull().any(axis=1)].head(1))

data.replace(to_replace=np.nan,value=0)
import matplotlib.pyplot as plt

team=['CSK','KKR','DC','MI']

score=[146,184,157,175]

    

      

plt.bar(team,score,color=['green','red','green','green'])

plt.title('IPL')

plt.xlabel('TEAMS')

plt.ylabel('SCORE')
a = np.array([1,8,2,6,4,9])

b = np.array([1,3,6,5])



c = np.intersect1d(a,b) #Finding the common items



print("Common items are: ",c)

print("\n")

for i in b:

    for j in a:

        if i == j:

            a = a[a!=j] #removing the common items from the array "a"

print(" 1st array:",a)

print("\n")

print(" 2nd array:",b)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



train = pd.read_csv("../input/iris/Iris.csv")





X = train.drop("Species",axis=1)

y = train["Species"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print("F1 Score:",f1_score(y_test, predictions,average='weighted'))

 

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)