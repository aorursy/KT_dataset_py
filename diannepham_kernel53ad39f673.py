import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, r2_score  

from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_score, cross_val_predict

from sklearn.decomposition import PCA

from math import sqrt

from sklearn.ensemble import RandomForestClassifier,  BaggingClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import sklearn

import time

from sklearn.datasets import make_classification

from sklearn.decomposition import PCA

from sklearn import metrics

from sklearn.metrics import f1_score

import os
print('Directory Path where files are located')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.getcwd() 

print(os.getcwd())

#Validate Current Path and create Path to data

from pathlib import Path

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)



#Import CSV into Pandas dataframe and test shape of file 

train = pd.read_csv(INPUT/"train.csv")

train.head(3)

train.shape
#Split into train and validation prior to cross validation

X_mnist_train, X_validation, y_mnist_train, y_validation = train_test_split(train.drop(['label'], axis = 1), 

                                                                            train['label'], train_size=30000, 

                                                                            test_size=12000, random_state=100)

print(X_mnist_train.shape)

print(X_validation.shape)

print(y_mnist_train.shape)

print(y_validation.shape)



#Split Train and Test

X_train, X_test, y_train, y_test = train_test_split(X_mnist_train, y_mnist_train, train_size = 0.7,

                                                    test_size =0.3, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17,5))



sns.countplot(train["label"], ax = ax1).set_title('Full Set')

sns.countplot(y_train, ax = ax2).set_title('Training Set')

sns.countplot(y_validation, ax = ax3).set_title('Validation Set')



plt.show()
start_time = time.process_time()



rfc = RandomForestClassifier(n_estimators=90, n_jobs=-1, criterion='gini',

                             max_features='sqrt', oob_score=True,  bootstrap = True, random_state=1)



rfc= rfc.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(rfc.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rfc.score(X_test, y_test)))

print("Accuracy on validation set: {:.3f}".format(rfc.score(X_validation, y_validation)))



print(metrics.classification_report(rfc.predict(X_train), y_train))

print(metrics.classification_report(rfc.predict(X_test), y_test))

print(metrics.classification_report(rfc.predict(X_validation), y_validation))

end_time = time.process_time()

runtime = end_time - start_time

print(runtime)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17,5))



cm_trn = confusion_matrix(y_train, rfc.predict(X_train))

sns.heatmap(cm_trn.T, square=True, annot=True, fmt='d', cbar=False, ax = ax1)

ax1.set_xlabel('Actual label')

ax1.set_ylabel('Predicted label')

ax1.set_title("Training")



c_mat = confusion_matrix(y_test, rfc.predict(X_test))

sns.heatmap(c_mat.T, square=True, annot=True, fmt='d', cbar=False, ax = ax2)

ax2.set_xlabel('Actual label')

ax2.set_ylabel('Predicted label')

ax2.set_title("Testing")



cm_vld = confusion_matrix(y_validation, rfc.predict(X_validation))

sns.heatmap(cm_vld.T, square=True, annot=True, fmt='d', cbar=False, ax = ax3)

ax3.set_xlabel('Actual label')

ax3.set_ylabel('Predicted label')

ax3.set_title("Validation")



fig.savefig('MnistCM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)



plt.show()
pca = PCA().fit(train.drop(['label'], axis = 1))

pca_plt=plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.savefig('PCAEstimate.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
start_time = time.process_time() 



pca = PCA(n_components=0.95)

pca.fit(train.drop(['label'], axis = 1))



X_pca = pca.transform(train.drop(['label'], axis = 1))

print("Original shape: {}".format(str(train.drop(['label'], axis = 1).shape)))

print("Reduced shape: {}".format(str(X_pca.shape)))

end_time = time.process_time() 

runtime = end_time - start_time  

print(runtime)
X_pca_train, X_pca_vld, y_pca_train, y_pca_vld= train_test_split(X_pca, train['label'], train_size=21000, 

                                                             test_size=7000, random_state=10)

print(X_pca_train.shape)

print(X_pca_vld.shape)

print(y_pca_train.shape)

print(y_pca_vld.shape)



X_pca_trn, X_pca_tst, y_pca_trn, y_pca_tst = train_test_split(X_pca_train, y_pca_train, train_size = 0.7,

                                                    test_size =0.3, random_state=10)

print(X_pca_trn.shape)

print(X_pca_tst.shape)

print(y_pca_trn.shape)

print(y_pca_tst.shape)
start_time = time.process_time() 



rfc_pca = RandomForestClassifier(n_estimators=90, n_jobs=-1, criterion='gini',

                                 max_features='sqrt', oob_score=True,  bootstrap = True, random_state=1)



rfc_pca = rfc_pca.fit(X_pca_trn, y_pca_trn)

print("Accuracy on training set: {:.3f}".format(rfc_pca.score(X_pca_trn, y_pca_trn)))

print("Accuracy on test set: {:.3f}".format(rfc_pca.score(X_pca_tst, y_pca_tst)))

print("Accuracy on validation set: {:.3f}".format(rfc_pca.score(X_pca_vld, y_pca_vld)))

    

# Compare

print(metrics.classification_report(rfc_pca.predict(X_pca_trn), y_pca_trn))

print(metrics.classification_report(rfc_pca.predict(X_pca_tst), y_pca_tst))

print(metrics.classification_report(rfc_pca.predict(X_pca_vld), y_pca_vld))

end_time = time.process_time() 

runtime = end_time - start_time

print(runtime)
test = pd.read_csv(INPUT/'test.csv')
start_time = time.process_time()



y_pred = rfc.predict(test)



end_time = time.process_time() 

runtime = end_time - start_time

print(runtime)
X_pca_test = pca.transform(test)
start_time = time.process_time()



y_pca_pred = rfc_pca.predict(X_pca_test)



end_time = time.process_time() 

runtime = end_time - start_time

print(runtime)