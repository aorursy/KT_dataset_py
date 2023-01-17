import pandas as pd

import numpy as np

import pylab as py

import scipy.optimize as opt

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
df.columns
data = df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time','DEATH_EVENT']]

data["DEATH_EVENT"] = data["DEATH_EVENT"].astype("int")

data.head()
X = np.asanyarray(data[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']])

X[0:5]
y = np.asanyarray(data['DEATH_EVENT'])

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
print("Train Set:",X_train.shape,y_train.shape)

print("Test Set:",X_test.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C = 0.01, solver = "liblinear").fit(X_train,y_train)

LR
from sklearn import svm

clf = svm.SVC(kernel="sigmoid")

clf.fit(X_train,y_train)

y_hat = clf.predict(X_test)

y_hat[0:5]
yhat = LR.predict(X_test)

yhat
yhat_prob = LR.predict_proba(X_test)

yhat_prob
#Evaluation

from sklearn.metrics import confusion_matrix,classification_report

import itertools

def plot_confusion_matrix(cm,classes,

                         normalize = False,

                         title='Confusion Matrix',

                         cmap = plt.cm.Blues):

    if normalize:

        cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]

        print("After Normalization")

    else:

        print("Without Normalization")

    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation = True,color='white')

    plt.yticks(tick_marks,classes,rotation =True,color='white')

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max()/2

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment = "center",

                color = 'white' if cm[i,j]>thresh else "black")

        

    plt.tight_layout()

    plt.xlabel("Predicted",color='white',size=20)

    plt.ylabel("True",color='white',size=20)
from sklearn.metrics import f1_score

f1_score(y_test,yhat,average = "weighted")
f1_score(y_test,y_hat,average = "weighted")
cnf_matrix=confusion_matrix(y_test,yhat,labels=[0,1])

np.set_printoptions(precision = 2)

print(classification_report(y_test,yhat))

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['Survived(0)','Died(1)'],normalize=False,title='Confusion Matrix')
cnf_matrix=confusion_matrix(y_test,y_hat,labels=[0,1])

np.set_printoptions(precision = 2)

print(classification_report(y_test,yhat))

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['Survived(0)','Died(1)'],normalize=False,title='Confusion Matrix')