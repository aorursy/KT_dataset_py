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
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

df.describe()
df.info()
df['species'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
#Análise exploratória

sns.pairplot(df, hue = 'species')

plt.show()
#Sepal_width X species

ax= sns.boxplot(x = 'species', y = 'sepal_width', data = df)

plt.show()
#Sepal_length

ax= sns.boxplot(x = 'species', y = 'sepal_length', data = df)

plt.show()
df_label = df['species']

df_data = df.drop('species', axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_data, df_label, test_size = 0.2, random_state = 42)
from sklearn.svm import SVC
clf_svm = SVC()

clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))