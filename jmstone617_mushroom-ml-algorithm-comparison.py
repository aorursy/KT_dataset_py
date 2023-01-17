# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mushrooms = pd.read_csv("../input/mushrooms.csv", header=0)



# Basic investigation to get our heads around the data

print("#############################")

print(mushrooms.head())

print("#############################")

print(mushrooms.describe())
# We can't do much with the labeled data as it exists, so we can use LabelEncoder()

# to turn the labels into unique integer values

le = preprocessing.LabelEncoder()

for col in mushrooms.columns:

    mushrooms[col] = le.fit_transform(mushrooms[col])



# And let's reinspect the translated data

print("#############################")

print(mushrooms.head())
predictors = [x for x in mushrooms.columns if x != "class"]



algs = {

    "GaussianNB": GaussianNB(),

    "Linear Regression": LinearRegression(normalize=True),

    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),

    "Decision Tree": DecisionTreeClassifier(random_state=0),

    "Extra Trees": ExtraTreesClassifier(n_estimators=5),

    "Support Vector Machine": SVC()

}



train_x, test_x, train_y, test_y = train_test_split(mushrooms[predictors], mushrooms["class"],

test_size=0.2, random_state=36)



for name, alg in algs.items():

    alg.fit(train_x, train_y)

    score = alg.score(test_x, test_y)

    print(name + ": " + str(score))
# Let's visualize the importance of the features for the Extra Trees algorithm

forest = algs["Extra Trees"]

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")

columns = list(mushrooms.columns.values)

for f in range(train_x.shape[1]):

    print("%d. %s (%f)" % (f + 1, columns[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(train_x.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(train_x.shape[1]), indices)

plt.xlim([-1, train_x.shape[1]])

plt.show()