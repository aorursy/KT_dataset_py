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
a = np.array([5,3,8,2,1])
b = np.array([6,8,3,9,15])
print('checking if b views to same memory :',b.base is a)
print('checking if a views to same memory :',a.base is b)
print('Divisible by 3 in a :',a[a%3==0])
print('Divisible by 3 in b :',b[b%3==0])
print("Sorted array b :",np.sort(b))
print("Sum of elements in a :",sum(a))

# Question 2
import pandas as pd
data = pd.read_csv("../input/titanic/train_and_test2.csv")

data.dropna(axis=1, how='all')
print(data.head())
print(data.shape)

print("Mean of 50 samples :",data[:50].mean())

print("Mean of male data :",data[data['Sex']==1].mean())

data['Fare'].max()
# Question 3
import matplotlib.pyplot as pp
slices = [86,83,86,90,88]
subject = ['English','Maths','Science','History','Geography']
colors = ['c','m','g','r','b']
pp.pie(slices,labels=subject,colors=colors,startangle=90,shadow=True,explode=(0,0.1,0,0,0),autopct='%1.1f%%')
pp.title('Marks')
pp.show
# Question 4
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/iris/Iris.csv")

X = train
X = train.drop("Species",axis=1)
y = train["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)




print("F1 Score:",f1_score(y_test, predictions,average='weighted'))
 
print("\nConfusion Matrix(below):\n")
confusion_matrix(y_test, predictions)