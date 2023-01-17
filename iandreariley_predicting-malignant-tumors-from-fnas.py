# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import binarize

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv")
df.head()
df.describe()
del df['Unnamed: 32']

del df['id']
d = {'M':1, 'B':0}

y = df['diagnosis'].map(d).values

X = df[df.columns[1:31]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(y_train.shape, y_test.shape)
knn = KNeighborsClassifier()

k_values = range(1, 31)

weight_values = ['uniform', 'distance']

param_dict = {'n_neighbors':k_values, 'weights':weight_values}

grid = GridSearchCV(knn, param_dict, cv=10, scoring='accuracy')

grid.fit(X_train, y_train)
print(grid.best_score_)

print(grid.best_params_)
%matplotlib inline



uniform_scores = []

distance_scores = []

all_scores = grid.cv_results_['mean_test_score']

all_params = grid.cv_results_['params']



#split

for i in range(len(all_scores)):

    if all_params[i]['weights'] == 'uniform':

        uniform_scores.append(all_scores[i])

    else:

        distance_scores.append(all_scores[i])

        

# Plot

plt.plot(k_values, uniform_scores)

plt.xlabel('K')

plt.ylabel('Mean Validation Score')

plt.title("Validation Score by K value, Uniform Weights")

plt.grid(True)
plt.plot(k_values, distance_scores)

plt.xlabel('K')

plt.ylabel('Mean Validation Score')

plt.title("Validation Score by K value, Distance Weights")

plt.grid(True)
knn_dist = KNeighborsClassifier(n_neighbors=12, weights='distance')

knn_unif = KNeighborsClassifier(n_neighbors=14, weights='uniform')

knn_dist.fit(X_train, y_train)

knn_unif.fit(X_train, y_train)
def model_accuracy(model, X, y):

    y_pred = model.predict(X)

    return metrics.accuracy_score(y, y_pred)



acc_dist = model_accuracy(knn_dist, X_test, y_test)

acc_unif = model_accuracy(knn_unif, X_test, y_test)



print("Best distance-weighted model test accuracy:", acc_dist)

print("Best uniform-weighted model test accuracy:", acc_unif)
y_prob_dist = knn_dist.predict_proba(X_test)

y_prob_unif = knn_unif.predict_proba(X_test)



def plot_hist(y, title='', bins=10):

    plt.hist(y, bins=bins)

    plt.title(title)

    plt.grid(True)



plot_hist(y_prob_dist[:,1], title="Distance kNN Probability Scores")
plot_hist(y_prob_unif[:,1], title="Uniform kNN Probability Scores")
# A finer grained histogram

plot_hist(y_prob_dist[:,1], title="Distance Probability Scores", bins=20)
y_pred_dist = knn_dist.predict(X_test)

metrics.confusion_matrix(y_test, y_pred_dist)
# plot ROC

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_dist[:,1])

plt.plot(fpr, tpr)

plt.title("ROC curve for Distance-weighted kNN (k=12)")

plt.xlabel("False Positive Rate (1 - specificity)")

plt.ylabel("True Positive Rate")

plt.grid(True)
fpr
thresholds