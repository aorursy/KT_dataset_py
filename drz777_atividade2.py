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

import matplotlib.pyplot as plt

import seaborn as sn

sn.set()



# Any results you write to the current directory are saved as output.
cars = pd.read_csv("/kaggle/input/carsdata/cars.csv",na_values = ' ')

cars.head()
cars.columns = ['mpg', ' cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',

       'year', ' brand']
cars.describe()
cars = cars.dropna()

cars['cubicinches']=cars['cubicinches'].astype(int)

cars['weightlbs'] =cars['weightlbs'].astype(int)
#cars.columns = ['mpg', ' cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',

#       'year', ' brand']
#cars.describe()
X = cars.iloc[:,:7]

X.head()
X_array = X.values

X_array
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300,n_init=10, random_state=0)

    kmeans.fit(X_array)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Cars')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()
import sklearn.datasets as dt
dic = dt.load_digits()

dic.keys()
dic.data.shape
dic.images.shape
import matplotlib.pyplot as plt

plt.imshow(dic.images[200])
X = dic.data

y = dic.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)
y_train.shape

X_train.shape
y_test.shape

y_train.shape
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)

knn

model = knn.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_score = model.score(X_test, y_test)

y_pred

y_score
import sklearn.metrics

model = str(round(model.score(X_test,y_test) * 100, 2))+"%"

print("O modelo k-NN foi",model)