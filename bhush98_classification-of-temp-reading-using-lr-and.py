# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the csv file

df = pd.read_csv("/kaggle/input/temperature-readings-iot-devices/IOT-temp.csv")

# Printing the first 5 entries in the Dataframe

df.head()
# Getting the number of rows and columns

df.shape
# Getting the dimensions of the Dataframe

df.ndim
# Information regarding the dataset

df.info()
# Getting the unique values from the id columns

unique_id = df['id'].unique()

print(len(unique_id))
unique_room_id = df['room_id/id'].unique()

print(unique_room_id)
df.describe()
data = df.iloc[:,3:]

data.head()
data.shape
from sklearn.preprocessing import LabelEncoder

lec = LabelEncoder()

data['out/in'] = lec.fit_transform(data['out/in'])

data.head()
import seaborn as sns

sns.pairplot(data = data , hue = 'out/in')
import matplotlib.pyplot as plt

plt.scatter(data['temp'],data['out/in'])

plt.title('Temperatur Scatter Plot')

plt.xlabel('Temperature')

plt.ylabel('IN/OUT')

plt.show()
X = data['temp'].values

print(X[0:5])
Y = data['out/in'].values

print(Y[0:5])
X.shape

X = X.reshape(-1,1)

X.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler().fit(X)

X = sc.transform(X)

print(X[0:5])
from sklearn.model_selection import train_test_split

X_train , x_test , Y_train , y_test = train_test_split(X,Y,test_size = 0.3)

print(X_train[0:5])
print(len(X_train))
print(len(x_test))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver = 'liblinear')

classifier.fit(X_train,Y_train)
from sklearn.model_selection import cross_validate

results = cross_validate(classifier , X , Y , cv=5)

print(results)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred,y_test))
from sklearn.svm import SVC

classifier2 = SVC(gamma = 'auto')

classifier2.fit(X_train,Y_train)
SVM_results = cross_validate(classifier2 , X, Y , cv=5)

print(SVM_results)
y_pred_classifier2 = classifier2.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_pred_classifier2,y_test)