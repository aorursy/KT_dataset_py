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
datasix1 = pd.read_csv('/kaggle/input/human-gait-phase-dataset/data/GP1_0.6_marker.csv')

dataseven2 = pd.read_csv('/kaggle/input/human-gait-phase-dataset/data/GP1_0.7_marker.csv')

datasix1.head()
data1 = pd.DataFrame({'secim':np.zeros(12000)})

data2 = pd.DataFrame({'secim':np.ones(12000)})



data2 = pd.concat([dataseven2,data2],axis=1)

data1 = pd.concat([datasix1,data1],axis=1)



data = data1.append(data2,ignore_index=True)

data.head()
x = data.drop(["secim"],axis=1)

y = data.secim.values

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.svm import SVC



svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("acc of svm is :",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train,y_train)

print('accuracy of bayes in test data is :', nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print('Accuracy of dec tree in test data is:',dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100,random_state=1)

rf.fit(x_train,y_train)

print('Random Forest accuracy on test data is : ',rf.score(x_test,y_test))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("test accuracy for Log Regressin is  {}".format(lr.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3) #n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



print("k={} nn score:{}".format(3,knn.score(x_test,y_test)))