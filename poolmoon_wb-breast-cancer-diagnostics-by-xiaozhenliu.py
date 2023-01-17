print_result(y_train, y_test, pred_train, pred_test)
from sklearn.lda import LDA



clf = LDA()

clf.fit(X_train, y_train) # Train the learner

pred_train = clf.predict(X_train) # Predictions on the training set

pred_test = clf.predict(X_test) #The predictions on the test set
print_result(y_train, y_test, pred_train, pred_test)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.



# Read the input file and put the data to pandas' dataframe format.

df = pd.read_csv("../input/data.csv")
df.head()
print(df.shape)
df.drop('id', axis=1, inplace=True)

df.drop('Unnamed: 32', axis=1, inplace=True)

df['diagnosis'].replace({'M':1,'B':0},inplace=True)
import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))
# Import data preprocessing tools from scikit-learn

from sklearn import preprocessing
# Method 1: Hold-out

from sklearn.model_selection import train_test_split

train, test = train_test_split(df,test_size=0.4, random_state=0)
# we can check their dimension

print(train.shape)

print(test.shape)
# Method 2: Cross-validation



# K-Fold

from sklearn.cross_validation import KFold

kf = KFold(df.shape[0],n_folds=10)

for train, test in kf:

    print("Train set shape: {}, Test set shape:{}".format(train.shape, test.shape))
# Leave-One-Out

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()



# There will be too many output lines, so I write it to a file, you can view it under the output tab

f = open("LOO_output.txt","w") 

f.write("This is the output of your LOO commands\n")

for train, test in loo.split(df):

    f.write("Train set shape: {}, Test set shape:{}\n".format(train.shape, test.shape))

f.close() 
# Method 3: Bootstrapping

from sklearn.utils import resample

train = resample(df,random_state=0)

train.describe()
df.describe()
# Find the test set

# Not finished yet
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# Use stratify to split the train and test set

train, test = train_test_split(df, test_size=0.4, random_state=0, stratify=df.ix[:,0])

X_train = train.ix[:,1:]

y_train = train.ix[:,0]

X_test = test.ix[:,1:]

y_test = test.ix[:,0]



# Use the random forest classifier

clf = RandomForestClassifier(random_state=0)

clf.fit(X_train, y_train) # Train the learner

pred_train = clf.predict(X_train) # Predictions on the training set

pred_test = clf.predict(X_test) #The predictions on the test set
from sklearn import metrics
# TODO: Error and Accuracy

accuracy_train = metrics.accuracy_score(y_train, pred_train)

accuracy_test = metrics.accuracy_score(y_test, pred_test)

err_train = 1 - accuracy_train

err_test = 1 - accuracy_test



print("The train error is {}; the train accuracy is {}.".format(round(err_train,4), round(accuracy_train,4)))

print("The test error is {}; the test accuracy is {}.".format(round(err_test,4), round(accuracy_test,4)))
# TODO: Precision, Recall and F1 Score

# Hint: Also try different values for beta and find F_beta

precision_train = metrics.precision_score(y_train, pred_train)

recall_train = metrics.recall_score(y_train, pred_train)

precision_test = metrics.precision_score(y_test, pred_test)

recall_test = metrics.recall_score(y_test, pred_test)

print("The training precision is {}; the training recall is {}.".format(round(precision_train,4), round(recall_train,4)))

print("The testing precision is {}; the testing recall is {}.".format(round(precision_test,4), round(recall_test,4)))



F1_train = metrics.f1_score(y_train, pred_train)

Fp5_train = metrics.fbeta_score(y_train, pred_train, beta=0.5)

F2_train = metrics.fbeta_score(y_train, pred_train, beta=2)

F1_test = metrics.f1_score(y_test, pred_test)

Fp5_test = metrics.fbeta_score(y_test, pred_test, beta=0.5)

F2_test = metrics.fbeta_score(y_test, pred_test, beta=2)



print("The training F1 score is {}, F_0.5 is {}, F_2 is {}.".format(round(F1_train, 4), round(Fp5_train, 4), round(F2_train, 4)))

print("The Testing F1 score is {}, F_0.5 is {}, F_2 is {}.".format(round(F1_test, 4), round(Fp5_test, 4), round(F2_test, 4)))
# TODO: ROC and AUC

# For Random Forest Classifier

# Reference: http://blog.csdn.net/lixiaowang_327/article/details/53434744



y_pred_pro = clf.predict_proba(X_test)  

scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values  

fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)

roc_auc = metrics.roc_auc_score(y_test, scores)



import matplotlib.pyplot as plt

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Random Forest Classifier ROC')

plt.legend(loc="lower right")

plt.show()



# For SVC



from sklearn.svm import SVC

sclf = SVC()

scores = sclf.fit(X_train, y_train).decision_function(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)

roc_auc = metrics.roc_auc_score(y_test, scores)

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('SVC ROC')

plt.legend(loc="lower right")

plt.show()

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state=0)

lr.fit(X_train, y_train) # Train the learner

pred_train = lr.predict(X_train) # Predictions on the training set

pred_test = lr.predict(X_test) #The predictions on the test set
def print_result(y_train, y_test, pred_train, pred_test):

    accuracy_train = metrics.accuracy_score(y_train, pred_train)

    accuracy_test = metrics.accuracy_score(y_test, pred_test)

    err_train = 1 - accuracy_train

    err_test = 1 - accuracy_test

    F1_train = metrics.f1_score(y_train, pred_train)

    F1_test = metrics.f1_score(y_test, pred_test)

    print("The train error is {}; the train accuracy is {}.".format(round(err_train,4), round(accuracy_train,4)))

    print("The test error is {}; the test accuracy is {}.".format(round(err_test,4), round(accuracy_test,4)))

    print("The training F1 score is {}.".format(round(F1_train, 4)))

    print("The Testing F1 score is {}.".format(round(F1_test, 4)))
print_result(y_train, y_test, pred_train, pred_test)
from sklearn.lda import LDA



clf = LDA()

clf.fit(X_train, y_train) # Train the learner

pred_train = clf.predict(X_train) # Predictions on the training set

pred_test = clf.predict(X_test) #The predictions on the test set
print_result(y_train, y_test, pred_train, pred_test)