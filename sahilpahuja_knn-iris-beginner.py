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
df=pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
df.isnull().sum()# NO missing value
df.Species.value_counts()
# Converting labels into numerical values

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df.Species=le.fit_transform(df.Species)
#Dividing into features and target

X=df.drop(columns=['Species','Id'])

Y=df.Species
#Train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=1)
X.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=5)



#Fitting the model on train data

knn.fit(x_train,y_train)



#Calculating accuracy score

acc_train=accuracy_score(y_train,knn.predict(x_train))

acc_test=accuracy_score(y_test,knn.predict(x_test))

print(f'accuracy score on train is {acc_train}')

print(f'accuracy score on test is {acc_test}')
#Lets see results when n_neighbours is different

acc_train=[]

acc_test=[]

for i in range(2,10):

    print(f'For n_neighbours = {i}')

    knn1=KNeighborsClassifier(n_neighbors=i)

    

    #Fitting the model on train data

    knn1.fit(x_train,y_train)

    

   #Calculating accuracy score

    acc_train.append(accuracy_score(y_train,knn1.predict(x_train)))

    acc_test.append(accuracy_score(y_test,knn1.predict(x_test)))

    print(f'accuracy score on train is {accuracy_score(y_train,knn1.predict(x_train))}')

    print(f'accuracy score on test is {accuracy_score(y_test,knn1.predict(x_test))}')

    print()
import matplotlib.pyplot as plt

#plotting test and train score

plt.plot(range(2,10),acc_train,color='g')

plt.plot(range(2,10),acc_test,color='orange')