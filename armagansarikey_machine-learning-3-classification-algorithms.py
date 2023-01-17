# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data from csv file

data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
data.shape
data.info()
data.describe()
data.columns
# Correlation

data.corr()
# Independent variables

x = data.iloc[:, -3:]

x
# Dependent variable

y = data.iloc[:, 0:1]

y
# Dividing into the data as train and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
x_train
x_test
y_train
y_test
# Scaling of data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)  # training and transforming from x_train

X_test = sc.transform(x_test)    # only transforming from x_test
# Creation of model

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train)
# Prediction

y_pred = log_reg.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

print('*********************')

print('*********************')

print(294/330) # Accuracy

# As it is seen below, the model has been predicted 294 values correctly from 330 values. 
# Creation of model

# K = 5 (Default value of the algorithm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

knn.fit(X_train, y_train)
# Prediction 

y_pred2 = knn.predict(X_test)



# Confusion matrix

cm2 = confusion_matrix(y_test, y_pred2)

print(cm2)

print('*********************')

print('*********************')

print(288/330) # Accuracy

# As it is seen below, the model has been predicted 288 values correctly from 330 values. 
# Creation of model

from sklearn.svm import SVC

svc = SVC(kernel='linear') 

svc.fit(X_train, y_train)
# Prediction

y_pred3 = svc.predict(X_test)

y_pred3
# Confusion Matrix

cm3 = confusion_matrix(y_test, y_pred3)

print(cm3)

print('*********************')

print('*********************')

print(291/330) # Accuracy

# As it is seen below, the model has been predicted 291 values correctly from 330 values. 
# Here, Kernel has been chosen as "rbf".

svc2 = SVC(kernel='rbf') 

svc2.fit(X_train, y_train)



y_pred_3 = svc2.predict(X_test)

y_pred_3



cm_3 = confusion_matrix(y_test, y_pred_3)

print(cm_3)

print('*********************')

print('*********************')

print(292/330) # Accuracy

# Creation of model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)
# Prediction

y_pred4 = gnb.predict(X_test)

y_pred4
# Confusion matrix

cm4 = confusion_matrix(y_test, y_pred4)

print(cm4)

print('*********************')

print('*********************')

print(229/330) # Accuracy
# Creation of model

# In Multinomial naive bayes, input x_train must be non-negative.

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(x_train, y_train)
# Prediction

y_pred_4 = mnb.predict(x_test)

y_pred_4
# Confusion matrix

cm_4 = confusion_matrix(y_test, y_pred_4)

print(cm_4)

print('*********************')

print('*********************')

print(294/330) # Accuracy
# Creation of model

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train, y_train)
# Prediction

y_pred5 = dtc.predict(X_test)

y_pred5
# Confusion matrix

cm5 = confusion_matrix(y_test, y_pred5)

print(cm5)

print('*********************')

print('*********************')

print(271/330) # Accuracy
# Creation of model

# Criterion' s default value is "gini".

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')

rfc.fit(X_train, y_train)
# Prediction

y_pred6 = rfc.predict(X_test)

y_pred6
# Confusion matrix

cm6 = confusion_matrix(y_test, y_pred6)

print(cm6)

print('*********************')

print('*********************')

print(278/330) # Accuracy
from sklearn import metrics
# ROC Curve with Random Forest Classification

y_proba_6 = rfc.predict_proba(X_test)

y_proba_6
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba_6[:,1], pos_label='male')

print(y_test)

print(y_proba_6[:,1])

print('fpr')

print(fpr)

print('tpr')

print(tpr)
# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.show()