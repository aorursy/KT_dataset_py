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
import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../input/Iris.csv')

df.head()
# define colors

df['Color'] = np.NaN

df.loc[df['Species'] == 'Iris-setosa', 'Color'] = 'red'

df.loc[df['Species'] == 'Iris-versicolor', 'Color'] = 'green'

df.loc[df['Species'] == 'Iris-virginica', 'Color'] = 'blue'

df.head()
plt.style.use('seaborn-whitegrid')

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

fig.suptitle('Iris Species')



for iris in df['Species'].unique().tolist():

    df_iris = df[df['Species'] == iris]

    axs[0].scatter(df_iris['SepalLengthCm'], df_iris['SepalWidthCm'], color=df_iris['Color'], label=iris)

    axs[1].scatter(df_iris['PetalLengthCm'], df_iris['PetalWidthCm'], color=df_iris['Color'], label=iris)



axs[0].set_title('Sepal Size')

axs[0].legend()

axs[0].set_xlabel('Length')

axs[0].set_ylabel('Width')

axs[1].set_title('Petal Size')

axs[1].legend()

axs[1].set_xlabel('Length')

axs[1].set_ylabel('Width')
X = df[['PetalLengthCm', 'PetalWidthCm']]

y = df['Species']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
neighbors = list(range(1, 16))

accuracy = np.zeros(len(neighbors))

for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    accuracy[i] = knn.score(X_train, y_train)



plt.style.use('seaborn-whitegrid')

plt.title('Accuracy')

plt.xlabel('k')

plt.ylabel('accuracy')

plt.plot(neighbors, accuracy)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print('Accuracy:', knn.score(X_test, y_test))