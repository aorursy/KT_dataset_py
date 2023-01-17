# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/iris/Iris.csv")

df.head()
df.shape
df.info()
df.isnull().sum()
df_setosa = df.loc[df['Species']=='Iris-setosa']

df_virginica = df.loc[df['Species']=='Iris-virginica']

df_versicolor = df.loc[df['Species']=='Iris-versicolor']
plt.plot(df_setosa['SepalLengthCm'], np.zeros_like(df_setosa['SepalLengthCm']),'bo')

plt.plot(df_versicolor['SepalLengthCm'], np.zeros_like(df_versicolor['SepalLengthCm']),'ro')

plt.plot(df_virginica['SepalLengthCm'], np.zeros_like(df_virginica['SepalLengthCm']),'go')

plt.show()
sns.FacetGrid(df, hue='Species', size = 7).map(plt.scatter, 'PetalLengthCm', 'SepalLengthCm').add_legend()

plt.show()
sns.pairplot(df, hue='Species', size=5)
import sklearn
from sklearn.model_selection import train_test_split
x=df.drop(['Species'], axis=1)

y=df['Species']
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.66, random_state=1,stratify=y)
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier

knn = KNeighborsClassifier(n_neighbors = 3)

# Fit the classifier to the data

knn.fit(x_train,y_train)
#show first 5 model predictions on the test data

knn.predict(x_test)[0:5]
#check accuracy of our model on the test data

knn.score(x_test, y_test)