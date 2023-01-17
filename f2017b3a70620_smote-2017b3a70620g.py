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
df_train = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

df_test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
X_train = df_train.drop('target',axis = 1)
y_train = df_train['target']
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
lr1 = LogisticRegressionCV(penalty='l2', cv = 100, verbose=10)

lr1.fit(X_train_res, y_train_res.ravel())
import matplotlib.pyplot as plt
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
y_train_pre = lr1.predict(X_train)



cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)



print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))





class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')

plt.show()
tmp = lr1.fit(X_train_res, y_train_res.ravel())
y_pred_sample_score = tmp.decision_function(X_train)





fpr, tpr, thresholds = roc_curve(y_train, y_pred_sample_score)



roc_auc = auc(fpr,tpr)
roc_auc
y_pre = lr1.predict(df_test)
y_pre
y_pred_pd = pd.DataFrame(data=y_pre, columns = ["Y_Pred"])
y_pred_pd.head()
y_pred_pd.value_counts()
my_submission = pd.DataFrame({'id': df_test.id, 'target': y_pred_pd.Y_Pred})
my_submission.head()
my_submission.to_csv('submission.csv', index=False)