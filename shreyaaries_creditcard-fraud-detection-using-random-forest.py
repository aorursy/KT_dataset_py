import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt

data = pd.read_csv("../input/creditcard.csv")

data.head()
x=data['Class']

bins = 2

plt.hist(x,bins)

plt.show()
data.groupby('Class').size()

from sklearn import preprocessing

data['StAmnt']=preprocessing.scale(data['Amount'])





data=data.drop(['Amount','Time'],axis=1)





# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=DeprecationWarning)

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

y = data['Class'] 

X = data.drop(['Class'], axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number of rows in X_train dataset: ", X_train.shape)

print("Number of rows in y_train dataset: ", y_train.shape)

print("Number of rows in X_test dataset: ", X_test.shape)

print("Number of rows in y_test dataset: ", y_test.shape)
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=DeprecationWarning)

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

y = data['Class'] 

X = data.drop(['Class'], axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number of rows in X_train dataset: ", X_train.shape)

print("Number of rows in y_train dataset: ", y_train.shape)

print("Number of rows in X_test dataset: ", X_test.shape)

print("Number of rows in y_test dataset: ", y_test.shape)
print(sum(y_train==1))

print(sum(y_train==0))

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print(sum(y_train_res==1))

print(sum(y_train_res==0))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

#from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
models = []



models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DTCL', DecisionTreeClassifier()))

#models.append(('SVM', SVC()))

models.append(('GNB', GaussianNB()))

models.append(('SGD', SGDClassifier()))

models.append(('RF', RandomForestClassifier()))



#testing models



results = []

names = []



for name, model in models:

    kfold = KFold(n_splits=10, random_state=42,shuffle=True)

    cv_results = cross_val_score(model, X_train_res, y_train_res, cv=kfold, scoring='roc_auc')

    results.append(cv_results)

    names.append(name)

    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

    print(msg)
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
Rfc = RandomForestClassifier()

Rfc.fit(X_train_res,y_train_res.ravel())

y_train_pred = Rfc.predict(X_train.values)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_train,y_train_pred)

np.set_printoptions(precision=2)



print("Recall metric in the train dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
y_test_pred = Rfc.predict(X_test)



# Compute confusion matrix

cnf_matrix_te = confusion_matrix(y_test,y_test_pred)

 



print("Recall metric in test dataset: ", cnf_matrix_te[1,1]/(cnf_matrix_te[1,0]+cnf_matrix_te[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve,auc

# ROC CURVE

Rfc = RandomForestClassifier()

Rfc.fit(X_train_res,y_train_res.ravel())

y_pred_proba = Rfc.predict_proba(X_test)[:, 1]



fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_pred_proba)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()