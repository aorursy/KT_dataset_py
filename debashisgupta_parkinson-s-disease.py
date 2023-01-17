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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train_dataframe = pd.read_csv('../input/train_data.txt',sep=',',header=None)
train_dataframe.head()
# taking only the valid features as given instructions from the website

x_train=train_dataframe.iloc[:,1:27]
x_train.head() 
y_train =train_dataframe.iloc[:,-1]
test_dataframe = pd.read_csv('../input/test_data.txt',sep=',',header=None)
test_dataframe.head()
# taking the valid features

x_test = test_dataframe.iloc[:,1:27]

y_test = test_dataframe.iloc[:,-1]
x_test.head()
#here i tried stnadardisation not normalisation

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

feat_x_train= sc.fit_transform(x_train)

feat_x_test = sc.transform(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.metrics import f1_score
def train_classifier(clf,X_train,y_train):

    clf.fit(X_train,y_train)



def predict_labels(clf,features,target,name):

    

    y_pred = clf.predict(features)

    cm = confusion_matrix(target.values,y_pred)

    print(cm)

    score= accuracy_score(target.values,y_pred)

    print("Accuracy on {} set {}".format(name,score))

    return f1_score(target.values,y_pred,pos_label=1)

def train_predict(clf,X_train,y_train,X_test,y_test):

    

    train_classifier(clf,X_train,y_train)

    print("F1 score for training set: {:.4f}".format(predict_labels(clf,X_train,y_train,'Train')))

    print("F1 score for test set: {:.4f}".format(predict_labels(clf,X_test,y_test,'Test')))
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
print("Naive Bayes ")

train_predict(clf,feat_x_train,y_train,feat_x_test,y_test)
from sklearn.ensemble import RandomForestClassifier

classifierRand = RandomForestClassifier(n_estimators=25,criterion='entropy',random_state=25)

train_predict(classifierRand,feat_x_train,y_train,feat_x_test,y_test)
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt
print("Naive Bayes")

class_names=[1,0]

print("On Training")

disp = plot_confusion_matrix(clf, feat_x_train, y_train,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 )

print(disp.confusion_matrix)

plt.show()

print("On Testing")

disp = plot_confusion_matrix(clf, feat_x_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 )

print(disp.confusion_matrix)

plt.show()
print("Randomfores")

class_names=[1,0]

print("On Training")

disp = plot_confusion_matrix(classifierRand, feat_x_train, y_train,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 )

print(disp.confusion_matrix)

plt.show()

print("On Testing")

disp = plot_confusion_matrix(classifierRand, feat_x_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 )

print(disp.confusion_matrix)

plt.show()