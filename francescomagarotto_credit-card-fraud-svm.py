# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

import sklearn.svm as svm

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = 'd' 

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.describe()
frodi = df[df['Class'] == 1]

non_frodi = df[df['Class'] == 0]

obj = ('Fraud', 'Not fraud')

y_pos = np.arange(len(obj))

plt.bar(y_pos, [len(frodi.index), len(non_frodi.index)], align='center', alpha=0.5)

plt.xticks(y_pos, obj)

plt.ylabel('Number')

plt.title('Dataset')
dataframe_not_fraud_rebalanced = resample(non_frodi, replace=False, n_samples=len(frodi.index), random_state=396)

dataframe_not_fraud_rebalanced
obj = ('Fraud', 'Not fraud')

y_pos = np.arange(len(obj))

plt.bar(y_pos, [len(frodi.index), len(dataframe_not_fraud_rebalanced.index)], align='center', alpha=0.5)

plt.xticks(y_pos, obj)

plt.ylabel('Number')

plt.title('Rebalanced')
dataset = pd.concat([frodi, dataframe_not_fraud_rebalanced])

dataset.Class.value_counts()
y = dataset['Class']

y
X = dataset.drop(['Class'],axis =1)

X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svm_cla = svm.SVC(kernel='linear')
svm_cla.fit(X_train, y_train)
test_pred = svm_cla.predict(X_test)
cm = confusion_matrix(y_test, test_pred)

plot_confusion_matrix(cm, ['0','1']) 
print("Recall: ", recall_score(y_test, test_pred))

print("Precision: ", precision_score(y_test, test_pred))

print("Accuracy: ", accuracy_score(y_test, test_pred))