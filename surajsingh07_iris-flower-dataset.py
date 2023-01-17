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
# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
data.info()
data.describe()
data.Species.value_counts()
# Making the copy of the data so that real data do not get messed up

df = data.copy()
sns.pairplot(df.drop('Id', axis=1), hue='Species')
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['Species'] = encoder.fit_transform(df['Species'])
X = df.drop(['Id', 'Species'], axis=1)

y = df['Species']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.neighbors import KNeighborsClassifier



scores = []

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    scores.append(knn.score(X_test, y_test))
sns.set_style('whitegrid')

sns.lineplot(x=range(1,10), y=scores, marker='o',)
final_knn = KNeighborsClassifier(n_neighbors=5)

final_knn.fit(X_train, y_train)

pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='viridis', cbar=False, annot_kws={'size':15})

plt.title('confusion matrix', fontsize=20)

plt.show()

print('classification report: \n{}'.format(classification_report(y_test, pred)))