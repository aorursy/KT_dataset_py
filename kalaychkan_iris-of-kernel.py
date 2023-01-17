# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/iris/Iris.csv")

data.head()

X=data.iloc[:,1:5];X.head()

y=data.iloc[:,5];y.head()

#print ("X Data\n ", X)

#print ("\ny Data\n", y)

from sklearn.model_selection import train_test_split

X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=0.3)

X_train.head()

y_train.head()

print ("length_of_X_train: " ,len(X_train))

print ("length_of_X_validation: " ,len(X_validation))

print ("length_of_y_train: " ,len(y_train))

print ("length_of_y_validation: " ,len(y_validation))
#SVM

from sklearn import svm

clf=svm.LinearSVC()

print ("model\n",clf)

clf.fit(X_train,y_train)

X_validation.iloc[0].values.reshape(1,-1)

clf.predict(X_validation.iloc[0].values.reshape(1,-1))

y_validation.iloc[0]

y_validation.iloc[0]

predictions=clf.predict(X_validation)

print("predictions\n",predictions)

type(y_validation)

for i in range(0,len(y_validation)):

    print("comparison(validation-prediction): ",y_validation.iloc[i],predictions[i])

clf.score(X_validation,y_validation)

clf.score(X_train,y_train)
#Classification



from sklearn.metrics import classification_report

print(classification_report(y_validation,predictions))



import itertools

import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

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



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

#confussion Matrix

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_validation, predictions)

class_names = y.unique();class_names

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=class_names,

                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')
#Cross_val_score



from sklearn.model_selection import cross_val_score

clf=svm.LinearSVC()

cross_val_score(clf,data.iloc[:,1:5],data.iloc[:,5],cv=5)



#KNN

from sklearn.neighbors import KNeighborsClassifier

clf2=KNeighborsClassifier()

cross_val_score(clf2,data.iloc[:,1:5],data.iloc[:,5],cv=5)