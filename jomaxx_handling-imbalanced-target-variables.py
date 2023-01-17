%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

import itertools

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.preprocessing import normalize



plt.rcParams.update({'figure.max_open_warning': 0})

warnings.simplefilter(action='ignore', category=FutureWarning)



print(os.listdir("../input"))
cc = pd.read_csv('../input/creditcard.csv')
cc.head()
# Split the column names into features and target

feature_columns = cc.columns[:-1].values

target_column = cc.columns[-1:].values[0]
plt.figure(figsize=(16,8))

ax = sns.countplot(x=target_column, data=cc)
cc.groupby(by='Class')['Class'].count()
fig, axs = plt.subplots(feature_columns.size,1, figsize=(15, 6*feature_columns.size))

fig.subplots_adjust(hspace = .5, wspace=.001)



axs = axs.ravel()



for i,feature in enumerate(feature_columns):

    c0_data = cc[cc['Class'] == 0][feature]

    c1_data = cc[cc['Class'] == 1][feature]

    axs[i].set_title(feature)

    sns.kdeplot(c0_data, legend=False, color='g', ax=axs[i])

    sns.rugplot(c0_data, color='g', ax=axs[i])

    sns.kdeplot(c1_data, legend=False, color='r', ax=axs[i])

    sns.rugplot(c1_data, color='r', ax=axs[i])
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):



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

    plt.show()
# Just use some selected columns as features

X = cc[['V10', 'V11','V12']].values

y = cc[target_column].values



#Lets split the dataset into a 

skf = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5)



for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    #print("Fold {0} Train Target: {1} Test Target: {2}".format(i, np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True)))

    

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    preds = clf.predict(X_test)    

    

    cm = confusion_matrix(y_test,preds)

    scores = precision_recall_fscore_support(y_test,preds)

    print("UNBALANCED")

    

    print("Precision: {0} ".format(scores[0]))

    print("Recall: {0} ".format(scores[1]))

    print("F1: {0} ".format(scores[2]))

    plot_confusion_matrix(cm, ['correct','fraudulent'], title='LogisticRegression with default settings')

    

    

    print('-'*50)

    clf_balanced = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)

    preds = clf_balanced.predict(X_test)

    

    cm_balanced = confusion_matrix(y_test,preds)

    scores_balanced = precision_recall_fscore_support(y_test,preds)

    print("BALANCED")

    print("Precision: {0} ".format(scores_balanced[0]))

    print("Recall: {0} ".format(scores_balanced[1]))

    print("F1: {0} ".format(scores_balanced[2]))

    plot_confusion_matrix(cm_balanced, ['correct','fraudulent'], title='LogisticRegression with balanced classes')

    

    

    clf_weights = LogisticRegression(random_state=0, class_weight={0: 0.001, 1: 0.999}).fit(X_train, y_train)

    preds = clf_weights.predict(X_test)

    

    cm_weights = confusion_matrix(y_test,preds)

    scores_weights = precision_recall_fscore_support(y_test,preds)

    print("BALANCED")

    print("Precision: {0} ".format(scores_weights[0]))

    print("Recall: {0} ".format(scores_weights[1]))

    print("F1: {0} ".format(scores_weights[2]))

    plot_confusion_matrix(cm_weights, ['correct','fraudulent'], title='LogisticRegression with custom class weights')