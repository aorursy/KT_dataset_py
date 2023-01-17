# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# This kernel will be useful for users who wants to try logistics regression with 
'''
1. One vs One 
2. One vs Rest
3. L1 regularisation and PCA
4. Compare the performance between PCA and L1 method
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # to perform principal component analysis
import matplotlib.pyplot as plt # for visualisation
from sklearn.datasets import load_digits # digit data
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report # to get different scores
from sklearn.metrics import confusion_matrix # for confusion matrix
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
digits_data = load_digits()
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits_data.data[10:15], digits_data.target[10:15])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)

X_train, X_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, test_size=0.25, random_state=0)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

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
X_test_pca = pca.transform(X_test)

import matplotlib.pyplot as plt
cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100
plt.bar(range(k), cum_sum)
plt.title("Around 95% of variance is explained by the 28 features");

def all_classifier(X_train_pca, X_test_pca, y_train, y_test):
    
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # fit one vs one classifier
    ovo_fit = OneVsOneClassifier(LogisticRegression())
    ovo_fit.fit(X_train_pca,y_train)
    ovo_pred = ovo_fit.predict(X_test_pca)
    
    print('===================================================================')
    print('\nLogistic Regression Score for OVO: \n {}'.format(ovo_fit.score(X_test_pca, y_test)))
    print('\nLogistic Regression  Confusion Matrix for OVO: \n {}'.format(confusion_matrix(y_test, ovo_pred)))
    print('\nLogistic Regression  Classification Report for OVO: \n {}'.format(classification_report(y_test, ovo_pred)))
     
    ovr_fit = OneVsRestClassifier(LogisticRegression())
    ovr_fit.fit(X_train_pca,y_train)
    ovr_pred = ovr_fit.predict(X_test_pca)
    
    print('\n===================================================================')

    print('\nLogistic Regression Score for OVR: \n {}'.format(ovr_fit.score(X_test_pca, y_test)))
    print('\nLogistic Regression  Confusion Matrix for OVR: \n {}'.format(confusion_matrix(y_test, ovr_pred)))
    print('\nLogistic Regression  Classification Report for OVR: \n {}'.format(classification_report(y_test, ovr_pred)))
    
    multiLR = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
    multiLR.fit(X_train_pca,y_train)
    multiLR_pred = multiLR.predict(X_test_pca)

    print('\n===================================================================')
    print('\nMultinomial Logistic Regression Score : \n {}'.format(multiLR.score(X_test_pca, y_test)))
    print('\nMultinomial Logistic Regression  Confusion Matrix : \n {}'.format(confusion_matrix(y_test, multiLR_pred)))
    print('\nMultinomial Logistic Regression Classification Report : \n {}'.format(classification_report(y_test, multiLR_pred)))

all_classifier(X_train_pca,X_test_pca, y_train,y_test)
multiLR = LogisticRegression(multi_class='multinomial',solver ='saga',penalty='l1')
multiLR.fit(X_train,y_train)
multiLR_pred = multiLR.predict(X_test)

print('\n===================================================================')
print('\nMultinomial Logistic Regression Score with L1 regularisation: \n {}'.format(multiLR.score(X_test, y_test)))
print('\nMultinomial Logistic Regression  Confusion Matrix with L1 regularisation: \n {}'.format(confusion_matrix(y_test, multiLR_pred)))
print('\nMultinomial Logistic Regression  Classification Report with L1 regularisation: \n {}'.format(classification_report(y_test, multiLR_pred)))
