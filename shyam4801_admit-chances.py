import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression





from sklearn.metrics import confusion_matrix,auc,roc_auc_score

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score



%matplotlib inline
ad = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

ad.shape
ad.info()
sns.pairplot(ad)
ad.drop('Serial No.',axis=1,inplace=True)

ad['Research'] = ad.Research.astype('category')

ad.columns
ad['University Rating'] = ad['University Rating'].astype('category')

ad['SOP'] = ad.SOP.astype('category')

ad['LOR '] = ad['LOR '].astype('category')

ad['CGPA'] = ad.CGPA.astype('category')

ad['Chance of Admit '] = ad['Chance of Admit '].astype('category')



ad.info()
ad.describe()
sns.distplot(ad['GRE Score']);
sns.distplot(ad['CGPA']);
sns.distplot(ad['TOEFL Score']);


import math

a = np.matrix(ad['Chance of Admit '])

print(a)



a = np.where(a > 0.5, 1, 0)

print(a)

x = ad[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',

       'LOR ', 'CGPA', 'Research']]

y = ad['Chance of Admit ']

y = a.flatten()

y = pd.Series(y)

# y
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 123)
li = LinearRegression()

ln = LinearRegression()



li.fit(X_train,y_train)

ln.fit(X_train,y_train)



ypred = li.predict(X_test)
li.score(X_train,y_train)

print(li.score(X_test,y_test))

ln.score(X_test,y_test)
li.coef_
y = li.coef_*X_test+li.intercept_



X_train.nunique()
# ad['Chance of Admit ']
log = LogisticRegression()

log.fit(X_train,y_train)

log.predict_proba(X_test)

ypreds = log.predict(X_test)

accuracy_score(y_test,ypreds)
from sklearn import metrics

from sklearn.metrics import classification_report



print(ypreds.shape)



cm = metrics.confusion_matrix(y_test,ypreds)

print(cm)

y_test.shape
print(classification_report(y_test,ypreds))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)

kpred = knn.predict(X_test)
knn.score(X_test,y_test)
# Model complexity

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(X_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(X_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(X_test, y_test))
# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('Value VS Accuracy',fontsize=20)

plt.xlabel('Number of Neighbors',fontsize=20)

plt.ylabel('Accuracy',fontsize=20)

plt.xticks(neig)

plt.grid()

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
knn6 = KNeighborsClassifier(n_neighbors = 6)

knn6.fit(X_train,y_train)

pred6 = knn6.predict(X_test)
knn6.score(X_test,y_test)
from sklearn import metrics

from sklearn.metrics import classification_report



print(pred6.shape)



cm = metrics.confusion_matrix(y_test,pred6)

print(cm)

y_test.shape
ad.columns
plt.scatter(y_test,y_test.index)

plt.scatter(kpred,y_test.index)