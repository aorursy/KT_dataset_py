import numpy as np

# to read our csv file

import pandas as pd

# matplotlib.pyplot is used for making graphs

import matplotlib.pyplot as plt

%matplotlib inline

# to train our model

from sklearn.ensemble import RandomForestClassifier

# to match the predicted value and real value of the predicting variable

from sklearn import metrics
dataset = pd.read_csv('../input/wine-quality-selection/winequality-white.csv')
dataset.head(5)
dataset.info()
data_train = dataset.head(3000)

data_test = dataset.tail(1898)
# choose the columns you want to keep in training set

cols = ['fixed acidity',

 'volatile acidity',

 'citric acid',

 'residual sugar',

 'chlorides',

 'free sulfur dioxide',

 'total sulfur dioxide',

 'density',

 'pH',

 'sulphates',

 'alcohol']

X_train=data_train[cols]

y_train=data_train.quality

X_test=data_test[cols]

y_test=data_test.quality
random_forest = RandomForestClassifier(n_estimators=50)  

random_forest.fit(X_train,data_train.quality)

# generate predictions on the training set

y_train_pred = random_forest.predict(X_train)



# generate predictions on the test set

y_test_pred = random_forest.predict(X_test)



metrics.accuracy_score(y_train,y_train_pred)
# calculate the accuracy of predictions on test data set

metrics.accuracy_score(y_test, y_test_pred)
plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

train_accuracies = [0]

test_accuracies = [0]

# iterate over a n_estimators values

for i in range(1, 25):

    random_forest = RandomForestClassifier(n_estimators=i)

    random_forest.fit(X_train, y_train)

    y_train_pred = random_forest.predict(X_train)

    y_test_pred = random_forest.predict(X_test)

    train_accuracy = metrics.accuracy_score(data_train.quality, y_train_pred)

    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)

    # append accuracies

    train_accuracies.append(train_accuracy)

    test_accuracies.append(test_accuracy)

# create two plots using matplotlib

plt.figure(figsize=(10, 5))

plt.plot(train_accuracies, label="training accuracy")

plt.plot(test_accuracies, label="test accuracy")

plt.legend(loc="upper left", prop={'size': 13})

plt.xticks(range(0, 26, 5))

plt.xlabel("n_estimators", size=20)

plt.ylabel("accuracy", size=20)

plt.show()
