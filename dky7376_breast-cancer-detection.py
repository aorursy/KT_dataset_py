import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



data = pd.read_csv('../input/data.csv', index_col = 0)

plt.figure(figsize = (14,4))

sns.barplot(x = data['diagnosis'], y = data['radius_mean'])
train_path = '../input/data.csv'

train = pd.read_csv(train_path, index_col = 0)

train.drop('Unnamed: 32', axis = 1, inplace = True)  # drop the unnecessary column

y = train['diagnosis']                                # target 

train.drop('diagnosis', axis = 1, inplace = True)      # train

train.head()
train.isnull().sum()
from sklearn.preprocessing import StandardScaler



scale = StandardScaler()

scale.fit(train)

X = pd.DataFrame(scale.transform(train))

X.columns = train.columns        # after transformation columns are changed hence put it back.

X.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



k = range(5,50,2)

arr = []

for i in k:

    clf = KNeighborsClassifier(n_neighbors = i)

    score = cross_val_score(clf, X, y, cv = 5, n_jobs = 2, scoring = 'accuracy')

    arr.append(score.mean())

print(arr)
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize = (14,4))

sns.scatterplot(x = k, y = arr)

plt.xlabel('K')

plt.ylabel('Accuracy')
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8)

clf = KNeighborsClassifier(n_neighbors = 7)

clf.fit(X_train, y_train)

preds = clf.predict(X_valid)

print(classification_report(y_valid, preds))

confusion_matrix(y_valid, preds)
