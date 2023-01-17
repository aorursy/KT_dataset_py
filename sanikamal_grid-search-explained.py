import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sn

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

from sklearn.model_selection import GridSearchCV

np.set_printoptions(precision=2)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import data

data = pd.read_csv('/kaggle/input/breast-cancer-csv/breastCancer.csv')
data.head()
data.tail()
data.columns
data.info()
data = data.drop(['id'],axis=1) #Drop 1st column

data = data[data['bare_nucleoli'] != '?'] #Remove rows with missing data

data['class'] = np.where(data['class'] ==2,0,1) #Change the Class representation

data['class'].value_counts() #Class distribution
#Split data into attributes and class

X = data.drop(['class'],axis=1)

y = data['class']
#perform training and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#Dummy Classifier

# clf=DummyClassifier(strategy="most_frequent")

# clf.fit(X_train,y_train)

clf = DummyClassifier(strategy= 'most_frequent',random_state=42).fit(X_train,y_train)

y_pred = clf.predict(X_test)


#Distribution of y test

print('y actual : \n' +  str(y_test.value_counts()))

#Distribution of y predicted

print('y predicted : \n' + str(pd.Series(y_pred).value_counts()))
# Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred,labels=np.unique(y_pred))))


#Dummy Classifier Confusion matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))
#Function to plot intuitive confusion matrix

import itertools

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
# Plot non-normalized confusion matrix

plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - DummyClassifier')

a = plt.gcf()

a.set_size_inches(8,4)

plt.show()
#Logistic regression

clf = LogisticRegression(solver="lbfgs",random_state=42).fit(X_train,y_train)

y_pred = clf.predict(X_test)
# Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(precision_score(y_test,y_pred)))

print('Recall Score : ' + str(recall_score(y_test,y_pred)))

print('F1 Score : ' + str(f1_score(y_test,y_pred)))
#Logistic Regression Classifier Confusion matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))
cnf_matrix = confusion_matrix(y_test, y_pred)
# Plot non-normalized confusion matrix

plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - LogisticRegression')

a = plt.gcf()

a.set_size_inches(8,4)

plt.show()
#Grid Search

clf = LogisticRegression(solver='liblinear',random_state=42)

grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}

grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall',cv=5,iid=True)

grid_clf_acc.fit(X_train, y_train)



#Predict values based on new parameters

y_pred_acc = grid_clf_acc.predict(X_test)
# New Model Evaluation metrics 

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))

print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))

print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))

print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))
#Logistic Regression (Grid Search) Confusion matrix

confusion_matrix(y_test,y_pred_acc)
cnf_matrix = confusion_matrix(y_test, y_pred)
# Plot non-normalized confusion matrix

plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - Logistic Regression (Grid Search)')

a = plt.gcf()

a.set_size_inches(8,4)

plt.show()
# explicitly require this experimental feature

from sklearn.experimental import enable_hist_gradient_boosting  # noqa

# now you can import normally from ensemble

from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate=0.005,random_state=42).fit(X_train, y_train)

y_pred=clf.predict(X_test)
clf.score(X_test,y_test)
#Logistic Regression Classifier Confusion matrix

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))
cnf_matrix_hgbc = confusion_matrix(y_test, y_pred)
# Plot non-normalized confusion matrix

plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix_hgbc, classes=class_names,

                      title='Confusion matrix - HistGradientBoostingClassifier')

a = plt.gcf()

a.set_size_inches(8,4)

plt.show()