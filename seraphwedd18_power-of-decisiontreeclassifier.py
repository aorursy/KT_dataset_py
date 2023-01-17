import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import time

start_time = time.time()

import gc
train = pd.read_csv("../input/cs6601ai-spring20-assign4-bonus/kaggle_train_2020_spring.csv", header=None)

test = pd.read_csv("../input/cs6601ai-spring20-assign4-bonus/kaggle_test_2020_spring_unlabelled.csv", header=None)
display(train)
display(test)
X = train.drop([0], axis=1)

Y = train[0]
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(X.corr(), annot=True, linewidth=0.02)

fig=plt.gcf()

fig.set_size_inches(20,20)

plt.show()
corr = X.corr()

drop_cols = []

for col in X.columns:

    if sum(corr[col].map(lambda x: abs(x) > 0.1)) <= 1:

        drop_cols.append(col)

X.drop(drop_cols, axis=1, inplace=True)

display(X)
import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(X.corr(), annot=True, linewidth=0.02)

fig=plt.gcf()

fig.set_size_inches(20,20)

plt.show()
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=100)

clf.fit(X, Y)

clf.score(X, Y)
pred = clf.predict(test.drop([x-1 for x in drop_cols], axis=1))

sub = pd.DataFrame()

sub['# Id'] = [x for x in range(len(pred))]

sub['Class'] = [x for x in map(int, pred)]

sub.to_csv("kaggle_result.csv", index=False)

display(sub)
end_time = time.time()



print("Total time spent on running this kernel: %i seconds" %(end_time-start_time))