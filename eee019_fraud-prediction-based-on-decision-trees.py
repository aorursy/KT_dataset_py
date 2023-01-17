import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import  train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from sklearn.decomposition import PCA





from subprocess import check_output

fraud_data = pd.read_csv("../input/creditcard.csv")



X = fraud_data.ix[:, fraud_data.columns != 'Class'] 

y = fraud_data.ix[:, fraud_data.columns == 'Class'] 



### splitting data in training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



###looking at the data



fraud_data.head()
def plot_confusion_matrix(cm,normalize=False,

                              title='Confusion matrix',

                              cmap=plt.cm.Blues):

        """

        This function prints and plots the confusion matrix.

        Normalization can be applied by setting `normalize=True`.

        """

        classes=['Non-Fraud','Fraud']

        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)

        plt.colorbar()

        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes, rotation=0)

        plt.yticks(tick_marks, classes)



        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # print("Normalized confusion matrix")

        else:

            1  # print('Confusion matrix, without normalization')



        # print(cm)



        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('Predicted label')

        return ()
desctree=DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

pca= PCA()

X_train_dtree = pd.DataFrame(pca.fit_transform(X_train))

desctree.fit=desctree.fit(X_train, y_train)

Y_train_dtree = pd.DataFrame(desctree.predict(X_test))

cnf_matrix_dtree = confusion_matrix(y_test, Y_train_dtree)

plt.figure()

plot_confusion_matrix(cnf_matrix_dtree, title='Confusion matrix for a decision tree')

plt.show()
rforest= RandomForestClassifier(criterion='entropy')

rforest.fit=rforest.fit(X_train,y_train)

Y_train_rforest = pd.DataFrame(rforest.predict(X_test))

cnf_matrix_rforest = confusion_matrix(y_test, Y_train_rforest)

plt.figure()

plot_confusion_matrix(cnf_matrix_rforest, title='Confusion matrix for random forest')

plt.show()