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
from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head(10)
df.isna().sum()
df = df.drop(['salary'], axis = 1)

df.head()
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])

df['ssc_b'] = le.fit_transform(df['ssc_b'])

df['hsc_b'] = le.fit_transform(df['hsc_b'])

df['hsc_s'] = le.fit_transform(df['hsc_s'])

df['degree_t'] = le.fit_transform(df['degree_t'])

df['workex'] = le.fit_transform(df['workex'])

df['specialisation'] = le.fit_transform(df['specialisation'])

df['status'] = le.fit_transform(df['status'])
df.head(10)
df.describe()
plt.figure(figsize=(20, 20))

corr_mat = df.corr().round(2)

sns.heatmap(data=corr_mat, annot=True)
features = ['ssc_p', 'hsc_p', 'degree_p', 'workex', 'specialisation']
y = df['status']

for i in features:

  x = df[i]

  plt.xlabel(i)

  plt.ylabel("Placed or not")

  plt.scatter(x, y)

  plt.show()
X = df[features]

Y = df['status']
X_train, x_test, Y_train, y_test = train_test_split(X, Y, random_state=4, test_size=0.3)
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
svmModel = SVC()

svmModel.fit(X_train, Y_train)

svmModel.score(x_test, y_test)
y_svm_hat = svmModel.predict(x_test)
plot_roc_curve(svmModel, x_test, y_test)
plot_confusion_matrix(svmModel, x_test, y_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_svm_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))
lrModel = LogisticRegression()

lrModel.fit(X_train, Y_train)

lrModel.score(x_test, y_test)
y_lr_hat = lrModel.predict(x_test)
plot_roc_curve(lrModel, x_test, y_test)
plot_confusion_matrix(lrModel, x_test, y_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_lr_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))
scores = []

for i in range(1, 21):

  knnModel = KNeighborsClassifier(n_neighbors=i)

  knnModel.fit(X_train, Y_train)

  score = knnModel.score(x_test, y_test)

  scores.append(score)



max(scores)
errors = [(1 - x) for x in scores]



plt.figure(figsize=(12, 12))

plt.plot(range(1, 21), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value') 

plt.xlabel('K') 

plt.ylabel('Error Rate')

plt.show()
best_n = scores.index(max(scores)) + 1

knnModel = KNeighborsClassifier(n_neighbors=best_n)

knnModel.fit(X_train, Y_train)

y_knn_hat = knnModel.predict(x_test)
plot_roc_curve(knnModel, x_test, y_test)
plot_confusion_matrix(knnModel, x_test, y_test)
truePositive, falsePositive, trueNegative, falseNegative = perf_measure(np.asarray(y_test), np.asarray(y_knn_hat))

print("Precision is", (truePositive / (truePositive + falsePositive)))

print("Recall is", (truePositive / (truePositive + falseNegative)))

print("Specificity is", (trueNegative / (trueNegative + falsePositive)))

print("Accuracy is", ((truePositive + trueNegative) / (truePositive + falsePositive + falseNegative + trueNegative)))