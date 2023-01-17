import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

print(os.listdir("../input"))

data = pd.read_csv("../input/creditcard.csv")

data.head(10)
#Data Preprocessing

data['normalized_amt'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Amount'], axis=1)

data.head()
data = data.drop(['Time'], axis=1)

data.head()
#Split Data in features and labels

x = data.iloc[:, data.columns!= 'Class']

y = data.iloc[:, data.columns== 'Class']
#Split data in train, val and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train.values.ravel())
y_pred = decision_tree.predict(x_test)

decision_tree.score(x_test, y_test )
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='None',

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

    plt.xticks(tick_marks, classes, rotation = 45)

    plt.yticks(tick_marks, classes)

    

    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
y_test.head()
y_pred
cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)
y_pred1 = decision_tree.predict(x)

y_actual = pd.DataFrame(y)
y_pred1
y_actual.head()
cnf_matrix = confusion_matrix(y_actual, y_pred1.round())

print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, classes = [0,1])