# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import itertools

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
# 데이터 읽어오기

df = pd.read_csv('../input/BankNote_Authentication.csv')

df.head(5)
# 목표라벨: Class

# 0은 위조됨을 나타냄. 1은 합법임을 나타냄

df['class'].value_counts()

# 예측을 위해 두 클래스간 데이터가 균등한지 확
# features, 대상 변수 정의

y = df['class']

X = df.drop(columns = ['class'])



# 데이터를 test set과 train set으로 나눔

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# 이진 분류를 위해 Logistic Regression 을 사용하여 예측

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train, y_train) # fitting the model

y_pred = LR.predict(X_test) # 예측
# 평가 Evaluation



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()





# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['Forged','Authorized'],

                      title='Confusion matrix, without normalization')
# extracting true_positives, false_positives, True_negatives, False_negatives

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("True Negatives: ", tn)

print("False Positives: ", fp)

print("False Negatives: ", fn)

print("True Positives: ", tp)



# Accuracy (%)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)

print("Accuracy {:0.2f}%".format(Accuracy))
# Precision 

Precision  = tp/(tp+fp)

print("Precision  {:0.2f}".format(Precision))
# Recall

Recall = tp/(tp+fn)

print("Recall {:0.2f}".format(Recall))
# F1 Score

f1 = (2*Precision * Recall)/(Precision + Recall)

print("F1 Score {:0.2f}".format(f1))
# Fbeta score

def fbeta(precision, recall, beta):

    return((1+pow(beta,2))*precision*recall)/ (pow(beta,2)*precision + recall)



f2 = fbeta(Precision, Recall, 2)

f0_5 = fbeta(Precision, Recall, 0.5)



print("F2 {:0.2f}".format(f2))

print("\nF0.5{:0.2f}".format(f0_5))
# Specificity

Specificity = tn/(tn+fp)

print("Specificity {:0.2f}".format(Specificity))
# Roc

import scikitplot as skplt # to make things easy

y_pred_proba = LR.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_pred_proba)

plt.show()
