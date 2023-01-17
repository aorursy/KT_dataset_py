# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head(10)
df.describe()
df.isna().sum()
features = [x for x in df.columns if x != 'target']
y = df['target']

for i in features:

  x = df[i]

  plt.xlabel(i)

  plt.ylabel("Heart disease")

  plt.scatter(x, y)

  plt.show()
plt.figure(figsize=(15, 15))

corr_mat = df.corr().round(2)

sns.heatmap(data=corr_mat, annot=True)
selected_features = ['cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[selected_features]

Y = df['target']
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4)
def perf_measure(y_actual, y_hat):

    TP = 0

    FP = 0

    TN = 0

    FN = 0



    for i in range(len(y_hat)): 

        if y_actual[i]==y_hat[i]==1:

           TP += 1

        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:

           FP += 1

        if y_actual[i]==y_hat[i]==0:

           TN += 1

        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:

           FN += 1



    return(TP, FP, TN, FN)
from sklearn.neighbors import KNeighborsClassifier

scores = []

for i in range(1, 18):

  knnModel = KNeighborsClassifier(n_neighbors=i)

  knnModel.fit(X_train, Y_train)

  score = knnModel.score(x_test, y_test)

  scores.append(score)

max(scores)
errors = [(1 - x) for x in scores]

plt.figure(figsize=(8, 8))

plt.plot(range(1, 18), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value') 

plt.xlabel('K') 

plt.ylabel('Error Rate')

plt.show()
knnModel = KNeighborsClassifier(n_neighbors=12)

knnModel.fit(X_train, Y_train)

knnModel.score(x_test, y_test)
plot_roc_curve(knnModel, x_test, y_test)
plot_confusion_matrix(knnModel, x_test, y_test)
y_knn_hat = knnModel.predict(x_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_knn_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))
from sklearn.linear_model import LogisticRegression

lrModel = LogisticRegression(max_iter=1200)

lrModel.fit(X_train, Y_train)

lrModel.score(x_test, y_test)
plot_roc_curve(lrModel, x_test, y_test)
plot_confusion_matrix(lrModel, x_test, y_test)
y_lr_hat = lrModel.predict(x_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_lr_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

for i in selected_features:

  X[i] = scaler.fit_transform(np.asarray(X[i]).reshape(-1, 1))
scaled_x_train, scaled_x_test, scaled_y_train, scaled_y_test = train_test_split(X, Y, random_state=4, test_size=0.3)
newLrModel = LogisticRegression()

newLrModel.fit(scaled_x_train, scaled_y_train)
newLrModel.score(scaled_x_test, scaled_y_test)
plot_roc_curve(newLrModel, scaled_x_test, scaled_y_test)
plot_confusion_matrix(newLrModel, scaled_x_test, scaled_y_test)
y_new_lr_hat = newLrModel.predict(scaled_x_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(scaled_y_test), np.asarray(y_new_lr_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))
scores = []

for i in range(1, 8):

  newKnnModel = KNeighborsClassifier(n_neighbors=i)

  newKnnModel.fit(scaled_x_train, scaled_y_train)

  score = newKnnModel.score(scaled_x_test, scaled_y_test)

  scores.append(score)



max(scores)
errors = [(1 - x) for x in scores]

plt.figure(figsize=(8, 8))

plt.plot(range(1, 8), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value') 

plt.xlabel('K') 

plt.ylabel('Error Rate')

plt.show()
newKnnModel = KNeighborsClassifier(n_neighbors=6)

newKnnModel.fit(scaled_x_train, scaled_y_train)
y_new_knn_hat = newKnnModel.predict(scaled_x_test)
newKnnModel.score(scaled_x_test, scaled_y_test)
plot_roc_curve(newKnnModel, scaled_x_test, scaled_y_test)
plot_confusion_matrix(newKnnModel, scaled_x_test, scaled_y_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(scaled_y_test), np.asarray(y_new_knn_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))
for i in selected_features:

  plt.scatter(scaled_x_test[i], scaled_y_test, color='grey')

  plt.scatter(scaled_x_test[i], y_new_knn_hat, color='red')

  plt.xlabel(i)

  plt.ylabel("Predictions")

  plt.show()