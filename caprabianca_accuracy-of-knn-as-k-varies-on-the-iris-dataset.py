# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.getcwd())
df=pd.read_csv('../input/Iris.csv')
df.set_index(['Id'], inplace=True)
df.head()
df.shape
df.describe()
df.groupby('Species').size()
df["Species"].unique()
import matplotlib.pyplot as plt

df.plot(kind='box', subplots='True', layout=(2,2), sharex=False, sharey=False)

plt.show()
df.hist()

plt.show()
from pandas.plotting import scatter_matrix

scatter_matrix(df)

plt.show()
import seaborn as sb

sb.pairplot(df)
array = df.values

print(array[:3])
X=array[:,0:3]

Y=array[:,4]

validation_size=0.2

seed=7

from sklearn.model_selection import train_test_split

X_train, X_validation, Y_train, Y_validation=train_test_split(X,Y,test_size=validation_size, random_state=seed)

scoring='accuracy'
from sklearn.neighbors import KNeighborsClassifier



graph = []



for i in range (1, 121):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, Y_train)

    accuracy = knn.score(X_validation, Y_validation)

    # print("The accuracy of i = {} is {}".format(i, accuracy))

    graph.append(accuracy)

    

plt.plot(graph)

plt.ylabel('Accuracy')

plt.xlabel('K value')

plt.show()