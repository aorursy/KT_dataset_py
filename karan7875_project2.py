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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow



import warnings

warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv', header=0)

train.head()
train.info()
train.describe()
import seaborn as sns

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
train.hist()
sns.set_style('whitegrid')

sns.countplot(x='target',data=train,palette='RdBu_r')
dataset = pd.get_dummies(train, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
dataset.head()
y = dataset['target']

X = dataset.drop(['target'], axis = 1)
from sklearn.model_selection import cross_val_score

knn_scores = []

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,X,y,cv=10)

    knn_scores.append(score.mean())
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')

for i in range(1,21):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
knn_classifier = KNeighborsClassifier(n_neighbors = 12)

score=cross_val_score(knn_classifier,X,y,cv=10)
score.mean()