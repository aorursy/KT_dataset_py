# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import mnist # import MNIST data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## define function having all classifiers
def all_classifier(X_train_pca, X_test_pca, y_train, y_test):
    
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # fit one vs one classifier
    ovo_fit = OneVsOneClassifier(LogisticRegression())
    ovo_fit.fit(X_train_pca,y_train)
    ovo_pred = ovo_fit.predict(X_test_pca)
    
    print('===================================================================')
    print('\nLogistic Regression Score OVO: \n {}'.format(ovo_fit.score(X_test_pca, y_test)))
    print('\nLogistic Regression  Confusion Matrix OVO: \n {}'.format(confusion_matrix(y_test, ovo_pred)))
    print('\nLogistic Regression  Classification Report OVO: \n {}'.format(classification_report(y_test, ovo_pred)))
     
    ovr_fit = OneVsRestClassifier(LogisticRegression())
    ovr_fit.fit(X_train_pca,y_train)
    ovr_pred = ovr_fit.predict(X_test_pca)
    
    print('\n===================================================================')

    print('\nLogistic Regression Score OVR: \n {}'.format(ovr_fit.score(X_test_pca, y_test)))
    print('\nLogistic Regression  Confusion Matrix OVR: \n {}'.format(confusion_matrix(y_test, ovr_pred)))
    print('\nLogistic Regression  Classification Report OVR: \n {}'.format(classification_report(y_test, ovr_pred)))
    
    multiLR = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
    multiLR.fit(X_train_pca,y_train)
    multiLR_pred = multiLR.predict(X_test_pca)

    print('\n===================================================================')
    print('\nLogistic Regression Score Multinomial: \n {}'.format(multiLR.score(X_test_pca, y_test)))
    print('\nLogistic Regression  Confusion Matrix Multinomial: \n {}'.format(confusion_matrix(y_test, multiLR_pred)))
    print('\nLogistic Regression  Classification Report Multinomial: \n {}'.format(classification_report(y_test, multiLR_pred)))

train = pd.read_csv('../input/train.csv')
Y = train['label']
train.drop(['label'], inplace = True, axis = 1 )

test = pd.read_csv('../input/test.csv')

Y = Y.values
X = train.values
X_test = test.values
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=0)

test = np.array(test, np.float64)

print('training data shape: {},{}'.format(X_train.shape, y_train.shape))
print('validation data shape: {},{}'.format(X_val.shape, y_val.shape))

## Normalize

X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

print('Every 5000th number {}'.format(y_train[::5000]))
pca=PCA() ## PCA
pca.fit_transform(X_train)

total=sum(pca.explained_variance_)
k=0
current_variance=0
while current_variance/total <= 0.95:
    current_variance += pca.explained_variance_[k]
    k=k+1
print(k)
pca = PCA(n_components=k)
X_train_pca=pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

import matplotlib.pyplot as plt
cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100
plt.bar(range(k), cum_sum)
plt.title("Around 95% of variance is explained by the 153 features");

print("Number transactions X_train dataset: ", X_train_pca.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_val_pca.shape)
print("Number transactions y_test dataset: ", y_val.shape)
# filter data

train_mask = np.isin(y_train, [0, 1, 2]) # subset data for 0, 1, 2 hand written digits
val_mask = np.isin(y_val, [0, 1, 2]) # # subset data for 0, 1, 2 hand written digits

X_train_pca_f, y_train_f = X_train_pca[train_mask], y_train[train_mask]
X_val_pca_f, y_val_f = X_val_pca[val_mask], y_val[val_mask]
# fit logistic modes
all_classifier(X_train_pca_f,X_val_pca_f, y_train_f,y_val_f)
all_classifier(X_train_pca,X_val_pca, y_train,y_val)