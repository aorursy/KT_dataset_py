import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as pl

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.utils import shuffle

from sklearn.metrics import roc_curve, auc

random_state = np.random.RandomState(0)
data = pd.read_csv('../input/handson-pima/Hands on Exercise Feature Engineering_ pima-indians-diabetes (1).csv')

data.head()
data.info()
#Extracting all the values in the data as array

array = data.values

X = array[:, 0:8]

y = array[:,8]
#Prepare train, test data

#Apply 2 algorithms Logistic Regression, SVM





X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.50, random_state = 1)

classifier1 = LogisticRegression()

classifier2 = svm.SVC(kernel = 'linear', probability = True)

probas1_ = classifier1.fit(X_train, y_train).predict_proba(X_test) #This will be help to detect probability

probas2_ = classifier2.fit(X_train,y_train).predict_proba(X_test)
#Compute ROC curve and AOC for logistic



fpr1,tpr1, thresholds1 = roc_curve(y_test, probas1_[:,1])

roc_auc1 = auc(fpr1,tpr1)

print("Area under ROC curve: %f" % roc_auc1)
#Compute ROC curve and AOC for SVM



fpr2,tpr2, thresholds2 = roc_curve(y_test, probas2_[:,1])

roc_auc2 = auc(fpr2,tpr2)

print("Area under ROC curve: %f" % roc_auc2)
#Plot the ROC Curve

#What to plot

pl.clf()

pl.plot(fpr1, tpr1, label = 'ROC Curve for logistic (area = %0.2f)' %roc_auc1)

pl.plot(fpr2, tpr2, label = 'ROC Curve for SVM (area = %0.2f)' %roc_auc2)

#How to plot

pl.plot([0,1],[0,1], 'k--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.0])

#Legends

pl.xlabel('False Positive Rate')

pl.ylabel('True Positive Rate')

pl.title('Receiver rating characteristic example ')

pl.legend(loc = "lower right")

pl.show()
i = np.arange(len(tpr1)) # index for df

roc1 = pd.DataFrame({'fpr1' : pd.Series(fpr1, index=i),'tpr1' : pd.Series(tpr1, index = i), '1-fpr1' : pd.Series(1-fpr1, index = i), 'tf1' : pd.Series(tpr1 - (1-fpr1), index = i), 'thresholds1' : pd.Series(thresholds1, index = i)})

print(roc1.loc[(roc1.tf1-0).abs().argsort()[:1]])



i = np.arange(len(tpr2)) # index for df

roc2 = pd.DataFrame({'fpr2' : pd.Series(fpr2, index=i),'tpr2' : pd.Series(tpr2, index = i), '1-fpr2' : pd.Series(1-fpr2, index = i), 'tf2' : pd.Series(tpr2 - (1-fpr2), index = i), 'thresholds2' : pd.Series(thresholds2, index = i)})

print(roc2.loc[(roc2.tf2-0).abs().argsort()[:1]])
roc1.head()
roc2.head()