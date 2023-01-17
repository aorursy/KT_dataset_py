import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

import math

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Social_Network.csv')
df.info()

df.head()
print (df.isnull().sum())
x_features = ['Age', 'EstimatedSalary']

x_df = df[x_features]

y_df = df['Purchased']

X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3, random_state = 5)
X1 = np.c_[np.ones((x_df.shape[0])),x_df]

plt.scatter(X1[:,1],X1[:,2],marker='o',c=y_df)

plt.show()
# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting SVM to the Training set

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, Y_train)
# Predicting the Test set results

Y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

cm = confusion_matrix(Y_test, Y_pred)

tn , fp, fn, tp = cm.ravel()

print('true negative: '+str(tn))

print('false positive: '+str(fp))

print('false negative: '+str(fn))

print('true positive: '+str(tp))
accuracy = (tp+tn)/(tp+tn+fp+fn)

misclassification = 1-accuracy

precision = tp/(fp + tp)

prevelance = (tp + fn) / (tp+tn+fp+fn)

print('accuracy: '+str(accuracy))

print('misclassification: '+str(misclassification))

print('precision: '+str(precision))

print('prevelance: '+str(prevelance))
# Visualising the Training set results

X_set, y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('yellow', 'blue'))(i), label = j)

plt.title('SVM (Training set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()
# Visualising the Test set results

X_set, y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('yellow', 'blue'))(i), label = j)

plt.title('SVM (Test set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()