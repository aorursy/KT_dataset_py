import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.cm import rainbow

%matplotlib inline



from sklearn.datasets import fetch_openml

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv('../input/mnist_train.csv')

test_data = pd.read_csv('../input/mnist_test.csv')
print("Training data:")

print("Shape: {}".format(train_data.shape))

print("Total images: {}".format(train_data.shape[0]))



print("Testing data:")

print("Shape: {}".format(test_data.shape))

print("Total images: {}".format(test_data.shape[0]))
train_y = train_data['label']

train_X = train_data.drop(columns = ['label'])



test_y = test_data['label']

test_X = test_data.drop(columns = ['label'])
train_labels = train_y.value_counts()

plt.figure(figsize = (12, 8))

cmap = rainbow(np.linspace(0, 1, train_labels.shape[0]))

plt.bar(train_labels.index.values, train_labels, color = cmap)

plt.xticks(train_labels.index.values)

plt.xlabel('Digits')

plt.ylabel('Count of images')

plt.title('Count of images for each digit (0 - 9)')
np.random.seed(0)

plt.figure(figsize = (20, 8))

for i in range(10):

    index = np.random.randint(train_X.shape[0])

    image_matrix = train_X.iloc[index].values.reshape(28, 28)

    plt.subplot(2, 5, i+1)

    plt.imshow(image_matrix, cmap=plt.cm.gray)
random_forest_classifier = RandomForestClassifier()

random_forest_classifier.fit(train_X, train_y)
pred_y = random_forest_classifier.predict(test_X)
print("Accuracy: {}%".format(accuracy_score(test_y, pred_y)*100))

print("Confusion Matrix:")

print("{}".format(confusion_matrix(test_y, pred_y)))
np.random.seed(0)

plt.figure(figsize = (20, 8))

for i in range(10):

    index = np.random.randint(test_X.shape[0])

    image_matrix = test_X.iloc[index].values.reshape(28, 28)

    plt.subplot(2, 5, i+1)

    plt.imshow(image_matrix, cmap=plt.cm.gray)

    plt.title("Model predicted number: {}".format(random_forest_classifier

                                                  .predict(test_X.iloc[index].values.reshape(1, -1))[0]))
from sklearn.model_selection import GridSearchCV



param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [10, 50, 100],

    'min_samples_split': [2, 4],

    'max_features': ['sqrt', 'log2']

}



grid = GridSearchCV(random_forest_classifier, param_grid = param_grid, cv = 5, verbose = 5, n_jobs = -1)

grid.fit(train_X, train_y)
best_estiomator = grid.best_estimator_
best_pred_y = best_estiomator.predict(test_X)

print("Accuracy: {}%".format(accuracy_score(test_y, best_pred_y)*100))

print("Confusion Matrix:")

print("{}".format(confusion_matrix(test_y, best_pred_y)))