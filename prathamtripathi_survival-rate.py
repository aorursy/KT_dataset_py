import pandas as pd

import numpy as np

import pylab as py

import matplotlib.pyplot as plt

import scipy.optimize as opt

from sklearn import preprocessing

%matplotlib inline
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
# label encoding the data 

from sklearn.preprocessing import LabelEncoder 

he = LabelEncoder() 

train_data['Sex']= he.fit_transform(train_data['Sex'])
test_data['Sex']= he.fit_transform(test_data['Sex'])
test_data.isnull().sum()
train_data = train_data.dropna()
train_data.shape
test_data.shape
my_train = train_data[["Pclass", "Sex", "SibSp", "Parch"]]
X = np.asanyarray(my_train[["Pclass", "Sex", "SibSp", "Parch"]])

X[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
y = np.asanyarray(train_data["Survived"])

y[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.5, random_state = 1)
print("Training Set: ", X_train.shape, y_train.shape)

print("Testing Set: ", X_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C = 0.01, solver = "liblinear").fit(X_train,y_train)

LR
y_hat = LR.predict(X_test)

y_hat
y_hat_prob = LR.predict_proba(X_test)

y_hat_prob
#evaluation by Jaccard Index

from sklearn.metrics import jaccard_score

jaccard_score(y_test,y_hat)
from sklearn.metrics import classification_report,confusion_matrix

import itertools

def plot_confusion_matrix(cm,classes,

                        normalize=True,

                        title='confusion matrix',

                        cmap=plt.cm.Blues):

    if normalize:

        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

        print("The normalized Confusion matrix")

    else:

        print("Without Normalization")

    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title,color='white')

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation=False,color='white',size=15)

    plt.yticks(tick_marks,classes,rotation=True,color='white',size=15)

    tmt='.2f'if normalize else 'd'

    thresh=cm.max()/2

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],tmt),

        horizontalalignment='center',

        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()

    plt.ylabel("True Label",color='white',size=20)

    plt.xlabel("False Label",color='white',size=20)

print(confusion_matrix(y_test,y_hat,labels=[1,0]))
cnf_matrix=confusion_matrix(y_test,y_hat,labels=[1,0])

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['Died = 0','Survived = 1'],normalize=True,title='Confusion Matrix')
#log loss

from sklearn.metrics import log_loss

log_loss(y_test,y_hat_prob)
print(classification_report(y_test,y_hat))
X_test_data = test_data[["Pclass", "Sex", "SibSp", "Parch"]]

pred = LR.predict(X_test_data)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})

output.to_csv('submission.csv', index=False)