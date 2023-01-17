import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.cm import rainbow

%matplotlib inline

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale

import xgboost as xgb

from xgboost.sklearn import XGBClassifier





# Load train dataset

digits_train = pd.read_csv("../input/digit-recognizer/train.csv")



# Load test dataset

digits_test = pd.read_csv("../input/digit-recognizer/test.csv")
print("Training data:")

print("Shape: {}".format(digits_train.shape))

print("Total images: {}".format(digits_train.shape[0]))



print("Testing data:")

print("Shape: {}".format(digits_test.shape))

print("Total images: {}".format(digits_test.shape[0]))
# Check for missing values in the train data

digits_train.isnull().sum().head(20)

# There are no missing values in the train data set.|
## Visualizing the number of class and counts in the datasets

plt.plot(figure = (22,16))

g = sns.countplot( digits_train["label"])

plt.title('Number of digit classes')

digits_train.label.astype('category').value_counts()
digits_train_subset = digits_train.loc[:8400]
## Visualizing the number of class and counts in the datasets

plt.plot(figure = (22,16))

g = sns.countplot( digits_train_subset["label"])

plt.title('Number of digit classes')

digits_train_subset.label.astype('category').value_counts()
X_tr = digits_train_subset.iloc[:,1:] # iloc ensures X_tr will be a dataframe

y_tr = digits_train_subset.iloc[:, 0]
#Lets split the train and test data with standard 70% train data and 30% test data

X_train, X_test, y_train, y_test = train_test_split(X_tr,y_tr,test_size=0.3, random_state=100, stratify=y_tr)
np.random.seed(0)

plt.figure(figsize = (20, 8))

for i in range(20):

    index = np.random.randint(X_train.shape[0])

    image_matrix = X_train.iloc[index].values.reshape(28, 28)

    plt.subplot(4, 5, i+1)

    plt.imshow(image_matrix, cmap=plt.cm.gray)
from time import time

random_forest_classifier = RandomForestClassifier()

start_time = time()

random_forest_classifier.fit(X_train, y_train)

print('Fitting time: ', time() - start_time)

y_pred = random_forest_classifier.predict(X_test)
print("Accuracy: {}%".format(accuracy_score(y_test, y_pred)*100))

print("Confusion Matrix:")

print("{}".format(confusion_matrix(y_test, y_pred)))
param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [10, 50, 100],

    'min_samples_split': [2, 4],

    'max_features': ['sqrt', 'log2']

}



start_time = time()

grid = GridSearchCV(random_forest_classifier, param_grid = param_grid, cv = 5, verbose = 5, n_jobs = -1)

grid.fit(X_train, y_train)

print('Fitting time: ', time() - start_time)

best_estimator = grid.best_estimator_
best_pred_y = best_estimator.predict(X_test)

print("Accuracy: {}%".format(accuracy_score(y_test, best_pred_y)*100))

print("Confusion Matrix:")

print("{}".format(confusion_matrix(y_test, best_pred_y)))