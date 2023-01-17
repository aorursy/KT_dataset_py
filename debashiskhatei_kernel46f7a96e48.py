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
import pandas as pd

import matplotlib.pyplot as plt

import re

import time

import warnings

import numpy as np

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import SMOTE

from collections import Counter

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold 

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")



from mlxtend.classifier import StackingClassifier



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression


data_variants = pd.read_csv('../input/training_variants')



data_text =pd.read_csv("../input/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
data_variants.head()
data_text.info()
data_text['LENGTH'] = data_text['TEXT'].str.len()
data_text.head()
data_new = pd.merge(left = data_variants, right = data_text,how='left',on= 'ID')
data_new.head()
sns.pairplot(data_new[data_new.isnull().any(axis = 1) == False],hue = 'Class')
data_new[data_new.isnull().any(axis = 1) == True]
data_new.loc[data_new['TEXT'].isnull(),'TEXT']  = data_new['Gene'] + " " + data_new['Variation']
data_new[data_new.isnull().any(axis = 1) == True]
data_new['LENGTH'] = data_new['TEXT'].str.len()
data_new.isnull().sum()
import re

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_processing(data_new):

    review = data_new

    review = re.sub('[^a-zA-Z0-9\n]',' ',review)

    review = review.lower()

    review = review.split()

    review = [word for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    return review

    
#data_new['TEXT1'] = data_new['TEXT'].apply(lambda x : text_processing(str(x)))
data_new.head()
# we create a output array that has exactly same size as the CV data

y_cv = [1,3,4,2,1,7,5,6,8,9,5]

cv_data_len = 11

cv_predicted_y = np.zeros((cv_data_len,9))

for i in range(cv_data_len):

    rand_probs = np.random.rand(1,9)

    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))
cv_predicted_y
data_new.head()
data_new['Gene'] = data_new['Gene'].str.replace(' ','_')
data_new['Variation'] = data_new['Variation'].str.replace(' ','_')
data_new.head(10)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_new,data_new['Class'],test_size = 0.2,stratify = data_new['Class'], random_state = 0)
X_train.groupby(['Class'])['Class'].value_counts().plot(kind = 'bar')
X_test.groupby(['Class'])['Class'].value_counts().plot(kind = 'bar')
X_train_val,X_test_val,y_train_val,y_test_val = train_test_split(X_train,y_train,test_size = 0.2, stratify = y_train, random_state = 0)
X_train_val.groupby(['Class'])['Class'].value_counts().plot(kind = 'bar')
X_test_val.groupby(['Class'])['Class'].value_counts().plot(kind = 'bar')
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
X_train.shape
X_train_val.shape
X_test.shape
X_test_val.shape
X_train.head()


#count_vec_gene = CountVectorizer(decode_error='ignore',stop_words='english',min_df=3,ngram_range=(1,1),analyzer='word',max_features=2000)

count_vec_gene = CountVectorizer()
X_train_val_gene_cv = count_vec_gene.fit_transform(X_train_val['Gene'])
X_train_val_gene_cv.shape
X_test_val_gene_cv = count_vec_gene.transform(X_test_val['Gene'])
X_test_val_gene_cv[:10][3]
X_test_gene_cv = count_vec_gene.transform(X_test['Gene'])
X_test_gene_cv[:10][3]


#count_vec_var = CountVectorizer(decode_error='ignore',stop_words='english',min_df=3,ngram_range=(1,1),analyzer='word',max_features=2000)

count_vec_var = CountVectorizer()
X_train_val_var_cv = count_vec_var.fit_transform(X_train_val['Variation'])
X_test_val_var_cv = count_vec_var.transform(X_test_val['Variation'])
X_test_var_cv = count_vec_var.transform(X_test['Variation'])
X_test_var_cv
#count_vec_text = CountVectorizer(decode_error='ignore',stop_words='english',min_df=3,ngram_range=(1,2),max_df = 0.8,analyzer='word')

count_vec_text = CountVectorizer(min_df=3)
X_train_val_text_cv = count_vec_text.fit_transform(X_train_val['TEXT'])
X_test_val_text_cv = count_vec_text.transform(X_test_val['TEXT'])
X_test_text_cv = count_vec_text.transform(X_test['TEXT'])
X_train_val_text_cv.shape
training_data_1 = hstack((normalize(X_train_val_gene_cv),normalize(X_train_val_var_cv)))
training_data_1
training_data_2 = hstack((training_data_1,normalize(X_train_val_text_cv)))
training_data_2
X_train_val.head()
from scipy import sparse
X_train_val_len = sparse.coo_matrix(X_train_val['LENGTH']).T
X_train_val_len
training_data_final = hstack((normalize(training_data_2),normalize(X_train_val_len)))
training_data_final
validation_data_1 = hstack((normalize(X_test_val_gene_cv),normalize(X_test_val_var_cv)))

validation_data_2 = hstack((validation_data_1,normalize(X_test_val_text_cv)))
X_test_val_len = sparse.coo_matrix(X_test_val['LENGTH']).T
X_test_val_len
validation_data_2
validation_data_final = hstack((validation_data_2,normalize(X_test_val_len)))
testing_data_1 = hstack((normalize(X_test_gene_cv),normalize(X_test_var_cv)))

testing_data_2 = hstack((testing_data_1,normalize(X_test_text_cv)))
X_test_len = sparse.coo_matrix(X_test['LENGTH']).T
testing_data_final = hstack((testing_data_2,normalize(X_test_len)))
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score

alphas = [0.0001,0.001,0.01,0.1,1,10,1000]

accuracy_array = []

log_loss_array = []

for i in alphas:

    classifier = MultinomialNB(alpha=i)    

    classifier.fit(training_data_final,y_train_val)

    sig_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    sig_clf.fit(training_data_final,y_train_val)

    y_pred = sig_clf.predict(validation_data_final)

    y_pred_prob = sig_clf.predict_proba(validation_data_final)

    accuracy_array.append(accuracy_score(y_test_val,y_pred))

    lg = log_loss(y_test_val,y_pred_prob,eps=1e-15)

    log_loss_array.append(lg)

    print('Validation Accuracy at alpha {} :'.format(i),accuracy_score(y_test_val,y_pred) )

    print('Validation Log Loss :', log_loss(y_test_val,y_pred_prob,eps=1e-15))

    cm = confusion_matrix(y_test_val, y_pred)

    #recall = np.diag(cm) / np.sum(cm, axis = 1)

    #precision = np.diag(cm) / np.sum(cm, axis = 0)

    #print('Precision :',precision_score(y_test_val, y_pred, average=None))

    #print('Recall :',np.mean(recall))

    
    classifier = MultinomialNB(alpha=0.001)    

    classifier.fit(training_data_final,y_train_val)

    sig_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    sig_clf.fit(training_data_final,y_train_val)

    y_pred = sig_clf.predict(testing_data_final)

    y_pred_prob = sig_clf.predict_proba(testing_data_final)

    print('Validation Accuracy at alpha {} :'.format(0.001),accuracy_score(y_test,y_pred) )

    print('Validation Log Loss :', log_loss(y_test,y_pred_prob,eps=1e-15))

    
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score

alphas = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,1000]

accuracy_array = []

log_loss_array = []

for i in alphas:

    classifier = SGDClassifier( alpha=i, penalty='l2', loss='log', random_state=42)

    classifier.fit(training_data_final,y_train_val)

    sig_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    sig_clf.fit(training_data_final,y_train_val)

    y_pred = sig_clf.predict(validation_data_final)

    y_pred_prob = sig_clf.predict_proba(validation_data_final)

    accuracy_array.append(accuracy_score(y_test_val,y_pred))

    lg = log_loss(y_test_val,y_pred_prob,eps=1e-15)

    log_loss_array.append(lg)

    print('Validation Accuracy at alpha {} :'.format(i),accuracy_score(y_test_val,y_pred) )

    print('Validation Log Loss :', log_loss(y_test_val,y_pred_prob,eps=1e-15))

    cm = confusion_matrix(y_test_val, y_pred)

    #recall = np.diag(cm) / np.sum(cm, axis = 1)

    #precision = np.diag(cm) / np.sum(cm, axis = 0)

    #print('Precision :',precision_score(y_test_val, y_pred, average=None))

    #print('Recall :',np.mean(recall))

    
    classifier = SGDClassifier(class_weight='balanced', alpha=1e-05, penalty='l2', loss='log', random_state=42)

    classifier.fit(training_data_final,y_train_val)

    sig_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    sig_clf.fit(training_data_final,y_train_val)

    y_pred = sig_clf.predict(testing_data_final)

    y_pred_prob = sig_clf.predict_proba(testing_data_final)

    print('Validation Accuracy at alpha {} :'.format(0.001),accuracy_score(y_test,y_pred) )

    print('Validation Log Loss :', log_loss(y_test,y_pred_prob,eps=1e-15))

    
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score

alphas = [100,300,1000,1500]

accuracy_array = []

log_loss_array = []

for i in alphas:

    classifier = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=3, random_state=42, n_jobs=-1)

    classifier.fit(training_data_final,y_train_val)

    sig_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    sig_clf.fit(training_data_final,y_train_val)

    y_pred = sig_clf.predict(validation_data_final)

    y_pred_prob = sig_clf.predict_proba(validation_data_final)

    accuracy_array.append(accuracy_score(y_test_val,y_pred))

    lg = log_loss(y_test_val,y_pred_prob,eps=1e-15)

    log_loss_array.append(lg)

    print('Validation Accuracy at alpha {} :'.format(i),accuracy_score(y_test_val,y_pred) )

    print('Validation Log Loss :', log_loss(y_test_val,y_pred_prob,eps=1e-15))

    cm = confusion_matrix(y_test_val, y_pred)

    #recall = np.diag(cm) / np.sum(cm, axis = 1)

    #precision = np.diag(cm) / np.sum(cm, axis = 0)

    #print('Precision :',precision_score(y_test_val, y_pred, average=None))

    #print('Recall :',np.mean(recall))

    
    classifier = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=10, random_state=42, n_jobs=-1)

    classifier.fit(training_data_final,y_train_val)

    sig_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    sig_clf.fit(training_data_final,y_train_val)

    y_pred = sig_clf.predict(testing_data_final)

    y_pred_prob = sig_clf.predict_proba(testing_data_final)

    print('Validation Accuracy at alpha {} :'.format(0.001),accuracy_score(y_test,y_pred) )

    print('Validation Log Loss :', log_loss(y_test,y_pred_prob,eps=1e-15))

    
clf1 = MultinomialNB(alpha=0.001)

clf1.fit(training_data_final,y_train_val)

sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")



clf2 = SGDClassifier(class_weight='balanced', alpha=1e-05, penalty='l2', loss='log', random_state=42)

clf2.fit(training_data_final,y_train_val)

sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")





clf3 = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=10, random_state=42, n_jobs=-1)

clf3.fit(training_data_final,y_train_val)

sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



sig_clf1.fit(training_data_final,y_train_val)

print("Naive Bayes :  Log Loss: %0.2f" % (log_loss(y_test_val, sig_clf1.predict_proba(validation_data_final))))

sig_clf2.fit(training_data_final,y_train_val)

print("Support vector machines : Log Loss: %0.2f" % (log_loss(y_test_val, sig_clf2.predict_proba(validation_data_final))))

sig_clf3.fit(training_data_final,y_train_val)

print("Random Forest : Log Loss: %0.2f" % (log_loss(y_test_val, sig_clf3.predict_proba(validation_data_final))))

print("-"*50)

alpha = [0.0001,0.001,0.01,0.1,1,10] 

best_alpha = 999

for i in alpha:

    lr = LogisticRegression(C=i)

    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

    sclf.fit(training_data_final,y_train_val)

    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(y_test_val, sclf.predict_proba(validation_data_final))))

    log_error =log_loss(y_test_val, sclf.predict_proba(validation_data_final))

    if best_alpha > log_error:

        best_alpha = log_error
clf1 = MultinomialNB(alpha=0.001)

clf1.fit(training_data_final,y_train_val)

sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")



clf2 = SGDClassifier(class_weight='balanced', alpha=1e-05, penalty='l2', loss='log', random_state=42)

clf2.fit(training_data_final,y_train_val)

sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")





clf3 = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=10, random_state=42, n_jobs=-1)

clf3.fit(training_data_final,y_train_val)

sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



sig_clf1.fit(training_data_final,y_train_val)

sig_clf2.fit(training_data_final,y_train_val)

sig_clf3.fit(training_data_final,y_train_val)

lr = LogisticRegression(C=0.1)

sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

sclf.fit(training_data_final,y_train_val)

print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(y_test, sclf.predict_proba(testing_data_final))))

print("Stacking Classifer : for the value of alpha: %f Accuracy: %0.3f" % (i, accuracy_score(y_test, sclf.predict(testing_data_final))))



    