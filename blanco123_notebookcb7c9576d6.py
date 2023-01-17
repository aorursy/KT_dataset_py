# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the dataset into a data table using Pandas

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.columns
dummies = pd.get_dummies(train, prefix='', prefix_sep='')

dummies = dummies.fillna(0)
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier



X, y = make_classification(n_samples=1460,

                           n_features=20,

                           n_informative=10,

                           n_redundant=0,

                           n_repeated=0,

                           n_classes=2,

                           random_state=0,

                           shuffle=False)



forest = RandomForestClassifier(n_estimators=250,

                                random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



print(importances)
# Plot the feature importances of the forest

plt.figure(figsize=(20,10))

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="g", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices,rotation=60)

plt.xlim([-1, X.shape[1]])

plt.show()
df = dummies.iloc[:, indices]

df = pd.concat([df, dummies['SalePrice']], axis=1, join_axes=[df.index])
df.columns
print(len(df))
from sklearn import metrics

from sklearn import linear_model

from sklearn.model_selection import train_test_split



# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(df.drop(['SalePrice'], axis=1),

                                                    df.SalePrice, test_size=0.3)



print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
# fit a model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
result = model.score(X_test, y_test)

print('Accuracy: ' ,result*100.0)
fig, ax = plt.subplots(figsize = (8,4))

plt.scatter(y_test, predictions)

ax.set_xlabel('Test values')

ax.set_ylabel('Predicted values')

plt.show()