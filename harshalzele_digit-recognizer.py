# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from time import time



import warnings

warnings.filterwarnings('ignore')
train_mnist=pd.read_csv('../input/train.csv')

test_mnist=pd.read_csv('../input/test.csv')
print(train_mnist.shape)

test_mnist.shape
train_mnist.describe()
train_mnist.info()
train_mnist.isnull().sum()/len(train_mnist.index)
train_mnist.sample(10)
X=train_mnist.drop(['label'],axis=1) # without lable column, we are creating X (independent (x-variable))

y=train_mnist['label'] # Only lable column, we are creating Y (Dependent (y-variable))
#Split the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, train_size=0.7, test_size=0.3, random_state = 100)



print('Training Data x-variable',X_train.shape)

print('Training Data y-variable',y_train.shape)

print('Testing Data x-variable',X_test.shape)

print('Testing Data y-variable',y_test.shape)
#MinMax Scaling

from sklearn.preprocessing import MinMaxScaler

min_max=MinMaxScaler()
# Fit and Tranformt he data for modelling purpose or we can use it for PCA

min_max.fit_transform(X_train,y_train)
#Reference https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#classification-of-text-documents-using-sparse-features



# #############################################################################



# Benchmark classifiers

def benchmark(clf):

    print('_' * 80)

    print("Training: ")

    print(clf)

    t0 = time() # Making note of current time.

    clf.fit(X_train, y_train) # Doing model fit

    train_time = time() - t0 # Subtracting the model fit start time to end time.

    print("train time: %0.3fs" % train_time)
from sklearn.linear_model import LogisticRegression

log_res = LogisticRegression(solver='sag', random_state=45)

#log_res.fit(X_train, y_train) # I am not using this now. Using as part of benchmark function.
benchmark(log_res)
y_pred=log_res.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score

from sklearn.utils.extmath import density
print("Dimensionality: %d" % log_res.coef_.shape[1])



print("Density: %f" % density(log_res.coef_))



print("Accuracy:   %0.3f" % accuracy_score(y_test, y_pred)) 



print('F1-Score (Macro) - ',f1_score(y_test, y_pred, average='macro'))



print('F1-Score (Micro) - ',f1_score(y_test, y_pred, average='micro'))



print('F1-Score (Weighted) - ',f1_score(y_test, y_pred, average='weighted'))



print("Classification Report\n%s\n" % (classification_report(y_test, y_pred)))   



print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))
from sklearn.naive_bayes import MultinomialNB

mul_nb = MultinomialNB()
benchmark(mul_nb)
y_pred_mul_nb=mul_nb.predict(X_test)
print("Dimensionality: %d" % mul_nb.coef_.shape[1])



print("Density: %f" % density(mul_nb.coef_))



print("Accuracy:   %0.3f" % accuracy_score(y_test, y_pred_mul_nb)) 



print('F1-Score (Macro) - ',f1_score(y_test, y_pred_mul_nb, average='macro'))



print('F1-Score (Micro) - ',f1_score(y_test, y_pred_mul_nb, average='micro'))



print('F1-Score (Weighted) - ',f1_score(y_test, y_pred_mul_nb, average='weighted'))



print("Classification Report\n%s\n" % (classification_report(y_test, y_pred_mul_nb)))   



print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred_mul_nb))
from sklearn.naive_bayes import BernoulliNB

bern_nb = BernoulliNB()
benchmark(bern_nb)
y_pred_bern_nb=bern_nb.predict(X_test)
print("Dimensionality: %d" % bern_nb.coef_.shape[1])



print("Density: %f" % density(bern_nb.coef_))



print("Accuracy:   %0.3f" % accuracy_score(y_test, y_pred_bern_nb)) 



print('F1-Score (Macro) - ',f1_score(y_test, y_pred_bern_nb, average='macro'))



print('F1-Score (Micro) - ',f1_score(y_test, y_pred_bern_nb, average='micro'))



print('F1-Score (Weighted) - ',f1_score(y_test, y_pred_bern_nb, average='weighted'))



print("Classification Report\n%s\n" % (classification_report(y_test, y_pred_bern_nb)))   



print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred_bern_nb))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
benchmark(rf)
y_pred_rf=rf.predict(X_test)
#print("Dimensionality: %d" % rf.coef_.shape[1])



#print("Density: %f" % density(rf.coef_))



print("Accuracy:   %0.3f" % accuracy_score(y_test, y_pred_rf)) 



print('F1-Score (Macro) - ',f1_score(y_test, y_pred_rf, average='macro'))



print('F1-Score (Micro) - ',f1_score(y_test, y_pred_rf, average='micro'))



print('F1-Score (Weighted) - ',f1_score(y_test, y_pred_rf, average='weighted'))



print("Classification Report\n%s\n" % (classification_report(y_test, y_pred_rf)))   



print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred_rf))
# Reference - https://scikit-learn.org/stable/modules/svm.html#svm



from sklearn.svm import LinearSVC

# Train a SVM classification model



linear_svc1=LinearSVC(random_state=45)
benchmark(linear_svc1)
y_pred_linear_svc1=linear_svc1.predict(X_test)
print("Dimensionality: %d" % linear_svc1.coef_.shape[1])



print("Density: %f" % density(linear_svc1.coef_))



print("Accuracy:   %0.3f" % accuracy_score(y_test, y_pred_linear_svc1)) 



print('F1-Score (Macro) - ',f1_score(y_test, y_pred_linear_svc1, average='macro'))



print('F1-Score (Micro) - ',f1_score(y_test, y_pred_linear_svc1, average='micro'))



print('F1-Score (Weighted) - ',f1_score(y_test, y_pred_linear_svc1, average='weighted'))



print("Classification Report\n%s\n" % (classification_report(y_test, y_pred_linear_svc1)))   



print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred_linear_svc1))
#importing PCA module

from sklearn.decomposition import PCA



# Initialising the PCA with svd_solver

pca=PCA(svd_solver='auto',random_state=45) 
pca.fit_transform(X_train)
#Making the screeplot - plotting the cumulative variance against the number of components

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
# 1) Initialising the PCA with svd_solver

#pca=PCA(svd_solver='randomized',random_state=45) 



# 2) Initialising the PCA with n_components, svd_solver and whiten

pca_100 = PCA(n_components=100, svd_solver='randomized', whiten=True)
# fit and tranform the x_train data into PCA. We transformed data for standardsing the data with StandardScaler in above steps. 

pca_100_array=pca_100.fit_transform(X_train)
# Double Checking number of components

len(pca_100.components_)
#creating correlation matrix for the principal components

corrmat = np.corrcoef(pca_100_array.transpose())
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (15,10))

sns.heatmap(corrmat,annot = True)
pca_100_X_train=pca_100.fit_transform(X_train)

pca_100_X_train.shape
# We did PCA transformation the x train data, now we will look into x-test data. We will only transform not fit

pca_100_X_test=pca_100.transform(X_test)
# We will use the same model which is used for evaluation (Random Forest Classifier)

rf_pca_100 = rf.fit(pca_100_X_train, y_train)
#Making prediction on the test data

pred_X_test = rf_pca_100.predict(pca_100_X_test)
accuracy_score(y_test, pred_X_test)
confusion_matrix(y_test,pred_X_test)
pca_09=PCA(n_components=0.9,whiten=True,random_state=54)
pca_09_X_train=pca_09.fit_transform(X_train)

len(pca_09.components_)
pca_09_X_test=pca_09.transform(X_test)
rf_pca_09=rf.fit(pca_09_X_train,y_train)
pred_09_X_test=rf_pca_09.predict(pca_09_X_test)
accuracy_score(y_test,pred_09_X_test)
confusion_matrix(y_test,pred_09_X_test)
#set ids and set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'ImageId' : y_test, 'Label': pred_09_X_test })

output.to_csv('submission.csv', index=False)