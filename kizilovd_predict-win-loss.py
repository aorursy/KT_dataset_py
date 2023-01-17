import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.model_selection import train_test_split

import numpy as np
frames = []

for i in range(1,20):

    frames.append(pd.read_csv("../input/fights_" + str(i) + ".csv"))

    

data = pd.concat(frames)

data.head()
data = data[data.Fight_Result != 'D']

data = data[data.Fight_Result != 'NC']

data.Fight_Result = data.Fight_Result.astype(int)

data_target = data['Fight_Result']
data_data = data[[

    'First KnockDowns',

    'First Significant Strikes',                  

    'First Total Strikes',

    'First Significant Strikes %',

    'First TakeDowns',

    'First TakeDowns Attempts',

    'First TD %',

    'First Subs',

    'First Passes',

    'First Rev.',

    'Second KnockDowns',    

    'Second Significant Strikes',                  

    'Second Total Strikes',    

    'Second Significant Strikes %',    

    'Second TakeDowns',

    'Second TakeDowns Attempts',    

    'Second TD %',    

    'Second Subs',    

    'Second Passes',    

    'Second Rev.']]
X = data_data

y = data_target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

clf = LogisticRegression(C=0.12, random_state=0, solver='lbfgs', max_iter=100000)



clf.fit(X_test, y_test)

clf.score(X, y)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt 

logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.figure(figsize=(10, 8))

plt.plot(fpr, tpr, label='Логистическая регрессия (площадь = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('1 - Специфичность')

plt.ylabel('Чувствительность')

plt.title('ROC-кривая')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from math import exp

coef = clf.coef_[0].tolist()



col = X_test.columns.values.tolist()

l = [coef, col]



for i in range(len(l[0])):

    print(l[1][i])

    print(l[0][i])

    print(exp(l[0][i]))
first_fighter = [0.000002, 80.997876,150.579212,55.565593,3.505367,8.462158,41.271576,0.966421,5.297134,0.0]

second_fighter = [1.200170,47.305305,99.308656,48.500755,0.499835,0.799860,17.994700,0.0,0.799760,0.100025]



X = first_fighter + second_fighter

c = clf.coef_ 

z = X @ c.T 



y = 1/(1 + np.exp(-z - clf.intercept_))

y
y_pred = clf.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
def plot_confusion_matrix(cm, classes,

                          normalize=True,

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

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



#     thresh = cm.max() / 2.

#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

#         plt.text(j, i, cm[i, j],

#                  horizontalalignment="center",

#                  color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)



# font = {'size' : 15}



# plt.rc('font', **font)



# cnf_matrix = confusion_matrix(y_test, clf.predict(X_test))

plt.figure(figsize=(10, 8))

plot_confusion_matrix(confusion_matrix, classes=['Loss', 'Win'],

                      title='Confusion matrix')

plt.savefig("conf_matrix.png")

plt.show()
from sklearn.metrics import brier_score_loss

print("Brier score of model: " + str(brier_score_loss(y_test, y_pred)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))