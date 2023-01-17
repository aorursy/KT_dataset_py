import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics#Import scikit-learn metrics module for accuracy calculation

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv", header=None)

train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv", header=None)

trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv", header=None)

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
print('train shape:', train.shape)

print('test shape:', test.shape)

print('trainLabel shape:', trainLabels.shape)

train.head()

train.info()
train.describe()
train.shape, test.shape, trainLabels.shape
# KNN with cross validation

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, train_test_split



X, y = train, np.ravel(trainLabels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Model complexity

neig = np.arange(1, 25)

kfold = 10

train_accuracy = []

val_accuracy = []

bestKnn = None

bestAcc = 0.0



# Loop over values of k

for i, k in enumerate(neig):

    #k from 1 to 25

    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit with knn

    knn.fit(X_train, y_train)

    #train accuracy

    train_accuracy.append(knn.score(X_train, y_train))

    #test accuracy

    val_accuracy.append(np.mean(cross_val_score(knn, X, y, cv=kfold)))

    if np.mean(cross_val_score(knn, X, y, cv=kfold)) > bestAcc:

        bestAcc = np.mean(cross_val_score(knn, X, y, cv=10))

        bestKnn = knn

        

print('Best Accuracy without feature scaling:', bestAcc)

print(bestKnn)
final_model = KNeighborsClassifier(n_neighbors = 3)

final_model.fit(train, trainLabels)

y_pred_knn=final_model.predict(X_test)

print("Training final: ", final_model.score(train, trainLabels))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))
final_test = final_model.predict(test)

final_test.shape
submission = pd.DataFrame(final_test)

print(submission.shape)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission
filename = 'London.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)