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
!pip install imbalanced-learn

import matplotlib.pyplot as plt

import seaborn as sns

import imblearn

from sklearn.preprocessing import StandardScaler, PowerTransformer

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

%matplotlib inline
df = pd.read_csv('../input/minordata/minor_data/train.csv') 
df.head(5)
corr = df.corr()

sns.heatmap(corr)
df.drop(['id', 'col_39'], axis=1, inplace = True) # thus from heatmap dropping col 39
X = df.drop(['target'], axis=1).values

y = df.target.values
norm = StandardScaler()

X = norm.fit_transform(X) # to make mean value 0 and standard deviation of 1
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_val.shape)

print("Number transactions y_test dataset: ", y_val.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE()

X_over, y_over = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_over.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_over.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_over==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_over==0)))

logReg = LogisticRegression(C=7, verbose=5, max_iter=1000)

logReg.fit(X_over, y_over)
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

y_train_pre = logReg.predict(X_train)



cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)



print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))



class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')

plt.show()
tmp = logReg.fit(X_train, y_train)
y_pred_sample_score = tmp.decision_function(X_val)





fpr, tpr, thresholds = roc_curve(y_val, y_pred_sample_score)



roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
roc_auc
tf = pd.read_csv('../input/minordata/minor_data/test.csv')

tf.drop(['id', 'col_39'], axis=1, inplace = True)

test = norm.fit_transform(tf)
predsprob = logReg.predict_proba(test)

ids = pd.read_csv('../input/minordata/minor_data/test.csv')

ids = ids["id"]

prob_zero = [item[0] for item in predsprob]

prob_one = [item[1] for item in predsprob]

outputprob = pd.DataFrame(data={"id" : ids, "target" : prob_one})
outputprob.to_csv('minor_temp_10.csv',index=False)