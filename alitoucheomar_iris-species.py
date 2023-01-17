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
import pandas as pd

data = pd.read_csv("../input/iris/Iris.csv")
data.shape
data.head(10)
data.info()
#train_test_split

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=.3 ,random_state= 40)

train.shape

x_train = train.iloc[:,1:5]

y_train = train.Species

x_test = test.iloc[:,1:5]

y_test = test.Species
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

matplotlib.rc('font', family='DejaVu Sans') 
# affichage des classes en fonction des variables sepal_length et sepal_width

sns.lmplot('SepalLengthCm', 'SepalWidthCm', data=data, hue='Species', fit_reg=False)

plt.show()
# affichage des classes en fonction des  variables petal_length et petal_width

sns.lmplot('PetalLengthCm', 'PetalWidthCm', data=data, hue='Species', fit_reg=False)

plt.show()
from sklearn.preprocessing import StandardScaler

#StandardScaler

standardscaler= StandardScaler()

standardscaler.fit(x_train)



x_train = pd.DataFrame(standardscaler.transform(x_train), columns=x_train.columns)

x_test = pd.DataFrame(standardscaler.transform(x_test), columns=x_test.columns)
from sklearn.svm import SVC

#SVM

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(x_train, y_train)



#The accuracy  

print('training data is {:.2f} out of 1'.format(svm.score(x_train, y_train)))

print('test data is {:.2f} out of 1'.format(svm.score(x_test, y_test)))
from sklearn.neighbors import KNeighborsClassifier

#KNN

knn = KNeighborsClassifier(n_neighbors=8, p=2, metric='minkowski')

knn.fit(x_train, y_train)



#The accuracy  

print('{:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))

print('{:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))
from sklearn.naive_bayes import GaussianNB

#naive_bayes

gnb = GaussianNB()

gnb.fit(x_train, y_train)



#The accuracy  

print(' {:.2f} out of 1 on training data'.format(gnb.score(x_train, y_train)))

print(' {:.2f} out of 1 on test data'.format(gnb.score(x_test, y_test)))