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
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df
df.describe()
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
df
df['diagnosis'].value_counts()
#We train the model, taking two features



X = df[['compactness_mean','area_mean']]

y = df[['diagnosis']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =1)
import seaborn as sns

sns.scatterplot(x='area_mean', y='compactness_mean', hue='diagnosis', data=X_test.join(y_test, how='outer') )
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
accuracies = pd.DataFrame(columns=['neighbors','accuracy'])

accuracies
#Loop to iterate through the number of neighbours. We need to find out which number gives the best accuracy!



for i in range(1, 10):

    

    knn = KNeighborsClassifier(n_neighbors = i, metric='euclidean')

    

    knn.fit(X_train, y_train.values.ravel())

    

    pred = knn.predict(X_test)

    

    acc = accuracy_score(pred, y_test)

    

    new_row = {'neighbors':i, 'accuracy':acc}

    

    accuracies = accuracies.append(new_row, ignore_index=True)
accuracies
#Plotting neighbours against accuracy



import matplotlib.pyplot as plt

plt.plot(accuracies.neighbors, accuracies.accuracy)