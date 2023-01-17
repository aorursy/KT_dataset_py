# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import re, nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_pickle('../input/515k-reviews-after-preprocessing/After_filling_Nans')
df = pd.read_pickle('../input/515k-reviews-after-preprocessing/After preprocessing')
# loading the positive reviews and negative reviews into text.

pos_reviews = data['Positive_Review'].values

pos_reviews = pos_reviews.tolist()

neg_reviews = data['Negative_Review'].values

neg_reviews = neg_reviews.tolist()

text = pos_reviews+neg_reviews
#converting the data into numpy arrays

summary = np.array(df.Summary)

score = df['score'].values
import time

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

start_time = time.time()

best_params = [] #store best parameters for MultinomialNB

parameters = {'alpha':[i for i in range(1,100,10)]} 

acc = []

score = list(score)

for i in range(2000,14000,1000):

    vec = CountVectorizer(max_features = i)

    data = vec.fit_transform(summary)

    nb = MultinomialNB()

    clf = GridSearchCV(nb, parameters,cv=5)

    x_train, x_test, y_train, y_test = train_test_split(data, score, test_size=0.3, random_state=42)

    clf.fit(x_train, y_train)

    acc.append(100.0*sum(clf.predict(x_test))/len((clf.predict(x_test))))

    best_params.append(clf.best_params_)

    vec = 0

    data = 0



print("--- %s seconds ---" % (time.time() - start_time))
##Confusion matrix

def show_confusion_matrix(C,class_labels=['0','1']):

    """

    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function

    class_labels: list of strings, default simply labels 0 and 1.



    Draws confusion matrix with associated metrics.

    """

    import matplotlib.pyplot as plt

    import numpy as np

    

    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

    

    # true negative, false positive, false negative, true positive

    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];



    NP = fn+tp # Num positive examples

    NN = tn+fp # Num negative examples

    N  = NP+NN # Total num of examples



    fig = plt.figure(figsize=(8,8))

    ax  = fig.add_subplot(111)

    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)



    # Draw the grid boxes

    ax.set_xlim(-0.5,2.5)

    ax.set_ylim(2.5,-0.5)

    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)

    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)

    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)

    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)



    # Set xlabels

    ax.set_xlabel('Predicted Label', fontsize=16)

    ax.set_xticks([0,1,2])

    ax.set_xticklabels(class_labels + [''])

    ax.xaxis.set_label_position('top')

    ax.xaxis.tick_top()

    # These coordinate might require some tinkering. Ditto for y, below.

    ax.xaxis.set_label_coords(0.34,1.06)



    # Set ylabels

    ax.set_ylabel('True Label', fontsize=16, rotation=90)

    ax.set_yticklabels(class_labels + [''],rotation=90)

    ax.set_yticks([0,1,2])

    ax.yaxis.set_label_coords(-0.09,0.65)





    # Fill in initial metrics: tp, tn, etc...

    ax.text(0,0,

            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(0,1,

            'False Neg: %d'%fn,

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(1,0,

            'False Pos: %d'%fp,

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))





    ax.text(1,1,

            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    # Fill in secondary metrics: accuracy, true pos rate, etc...

    ax.text(2,0,

            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(2,1,

            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(2,2,

            'Accuracy: %.2f'%((tp+tn+0.)/N),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(0,2,

            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))



    ax.text(1,2,

            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),

            va='center',

            ha='center',

            bbox=dict(fc='w',boxstyle='round,pad=1'))





    plt.tight_layout()

    plt.show()
start_time = time.time()

from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss



score_Log_reg = []

y_pred = clf.predict(x_test)

conf_NB = confusion_matrix(y_test, y_pred)    



print("Confusion matrix:\n",conf_NB)



#ROC for a given alpha for NB

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class

probs = clf.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



#Plot ROC

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#print the log loss

a = log_loss(y_test, probs)

print("The log loss for the Naive bayes is:",a)



#print confusion matrix

show_confusion_matrix(conf_NB,['Negative','Positive'])



#Precision and recall

tn = conf_NB[0,0]; fp = conf_NB[0,1]; fn = conf_NB[1,0]; tp = conf_NB[1,1];



precision = 100*float(tp)/(tp+fp)

recall = 100*float(tp)/(tp+fn)



print("Precision :",precision)

print("Recall :",recall)



tp = conf_NB[0][0]

tn = conf_NB[1][1]

print("The accuracy is {} %".format(round(100.0*(tp+tn)/len(y_test),2)))

print('------------ %s seconds ------------'%(time.time()-start_time))
#Logistic regression hyperparameter tuning

import warnings

from sklearn.linear_model import SGDClassifier

warnings.filterwarnings('ignore')

start_time = time.time()

best_params_logreg = []

parameters = {'loss' :['log'],'penalty':['l1','l2','elasticnet'],'alpha':[float(i)/10 for i in range(1,10,1)],'n_jobs':[-1]}

warnings.filterwarnings('ignore')

clf = SGDClassifier()

clf = GridSearchCV(clf, parameters,cv=5)

clf.fit(x_train, y_train)

best_params_logreg.append(clf.best_params_)

print('Best parameters for Logistic Regression are:',best_params_logreg)

print("--- %s seconds ---" % (time.time() - start_time))
clf = SGDClassifier(loss = 'log',penalty = 'l2',alpha = 0.1, n_jobs = -1)

#choose acc to best parameters

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

conf_log_ref = confusion_matrix(y_test, y_pred)



print("Confusion matrix:\n",conf_log_ref)

#ROC for a given hyperparameters for logistic regression

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class

probs = clf.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



#Plot ROC

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#print the log loss

a = log_loss(y_test, probs)

print("The log loss for the Logistic regression is:",a)



#print confusion matrix

show_confusion_matrix(conf_log_ref,['Negative','Positive'])



#Precision and recall

tn = conf_log_ref[0,0]; fp = conf_log_ref[0,1]; fn = conf_log_ref[1,0]; tp = conf_log_ref[1,1];



precision = 100*float(tp)/(tp+fp)

recall = 100*float(tp)/(tp+fn)



print("Precision :",precision)

print("Recall :",recall)



tp = conf_log_ref[0][0]

tn = conf_log_ref[1][1]

print("The accuracy is {} %".format(round(100.0*(tp+tn)/len(y_test),2)))
#SVM hyperparameter tuning

start_time = time.time()

best_params_SVM = []

parameters = {'loss' :['hinge'],'penalty':['l1','l2','elasticnet'],'alpha':[float(i)/10 for i in range(1,10,1)],'n_jobs':[-1]}

 

clf = SGDClassifier()

clf = GridSearchCV(clf, parameters,cv=5)

clf.fit(x_train, y_train)

best_params_SVM = clf.best_params_

print("Best hyperparameters for linear SVM:",best_params_SVM)

print('------{} seconds-------'.format(time.time()-start_time))
#Linear SVM

clf = SGDClassifier(penalty = 'l2', alpha = 0.1, n_jobs =  -1, loss = 'hinge')

#choose acc to best parameters

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

conf_SVM_ref = confusion_matrix(y_test, y_pred)

print("Confusion matrix:\n",conf_SVM_ref)

#print confusion matrix

show_confusion_matrix(conf_SVM_ref,['Negative','Positive'])

#Precision and recall

tn = conf_SVM_ref[0,0]; fp = conf_SVM_ref[0,1]; fn = conf_SVM_ref[1,0]; tp = conf_SVM_ref[1,1];

precision = 100*float(tp)/(tp+fp)

recall = 100*float(tp)/(tp+fn)

print("Precision :",precision)

print("Recall :",recall)

tp = conf_SVM_ref[0][0]

tn = conf_SVM_ref[1][1]

print("The accuracy is {} %".format(round(100.0*(tp+tn)/len(y_test),2)))
# Hyperparameter tuning for MultinomialNB with Bigrams

start_time = time.time()

best_params = []

parameters = {'alpha':[i for i in range(1,100,10)]}

features = [i for i in range(10000,130000,10000)]

acc = []

score = list(score)

for i in range(2000,14000,1000):

    vec = CountVectorizer(ngram_range=(1,2),max_features = i)

    data = vec.fit_transform(summary)

    nb = MultinomialNB()

    clf = GridSearchCV(nb, parameters,cv=5)

    x_train, x_test, y_train, y_test = train_test_split(data, score, test_size=0.3, random_state=42)

    clf.fit(x_train, y_train)

    acc.append(100.0*sum(clf.predict(x_test))/len((clf.predict(x_test))))

    best_params.append(clf.best_params_)

    vec = 0

    data = 0

print('-------- %s seconds -------'%(time.time()-start_time))
# MultinomialNb with Bigrams

score_Log_reg = []

y_pred = clf.predict(x_test)

conf_NB = confusion_matrix(y_test, y_pred)    



print("Confusion matrix:\n",conf_NB)



#ROC for a given alpha for NB

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class

probs = clf.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



#Plot ROC

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#print the log loss

a = log_loss(y_test, probs)

print("The log loss for the Naive bayes is:",a)



#print confusion matrix

show_confusion_matrix(conf_NB,['Negative','Positive'])



#Precision and recall

tn = conf_NB[0,0]; fp = conf_NB[0,1]; fn = conf_NB[1,0]; tp = conf_NB[1,1];



precision = 100*float(tp)/(tp+fp)

recall = 100*float(tp)/(tp+fn)



print("Precision :",precision)

print("Recall :",recall)



tp = conf_NB[0][0]

tn = conf_NB[1][1]

print("The accuracy is {} %".format(round(100.0*(tp+tn)/len(y_test),2)))
# Hyperparameter tuning for MultinomialNB with Bigrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

start_time = time.time()

best_params = []

parameters = {'alpha':[i for i in range(1,100,10)]}

acc = []

score = list(score)

for i in range(2000,14000,1000):

    vec = TfidfVectorizer(max_features = i)

    data = vec.fit_transform(summary)

    nb = MultinomialNB()

    clf = GridSearchCV(nb, parameters,cv=5)

    x_train, x_test, y_train, y_test = train_test_split(data, score, test_size=0.3, random_state=42)

    clf.fit(x_train, y_train)

    acc.append(100.0*sum(clf.predict(x_test))/len((clf.predict(x_test))))

    best_params.append(clf.best_params_)

    vec = 0

    data = 0

print('-------- %s seconds -------'%(time.time()-start_time))
# MultinomialNb with TF-IDF

from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss



score_Log_reg = []

y_pred = clf.predict(x_test)

conf_NB = confusion_matrix(y_test, y_pred)    



print("Confusion matrix:\n",conf_NB)



#ROC for a given alpha for NB

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class

probs = clf.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



#Plot ROC

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#print the log loss

a = log_loss(y_test, probs)

print("The log loss for the Naive bayes is:",a)



#print confusion matrix

show_confusion_matrix(conf_NB,['Negative','Positive'])



#Precision and recall

tn = conf_NB[0,0]; fp = conf_NB[0,1]; fn = conf_NB[1,0]; tp = conf_NB[1,1];



precision = 100*float(tp)/(tp+fp)

recall = 100*float(tp)/(tp+fn)



print("Precision :",precision)

print("Recall :",recall)



tp = conf_NB[0][0]

tn = conf_NB[1][1]

print("The accuracy is {} %".format(round(100.0*(tp+tn)/len(y_test),2)))
from gensim.models import word2vec

from gensim.models import KeyedVectors

start_time = time.time()

model = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

print('----------- %s seconds ------------'%(time.time()-start_time))
model.wv.similarity('dinner','lunch')
model.wv.most_similar('tasty')
model.wv.most_similar('comfortable')
model.wv.most_similar('london')
model.wv.most_similar('europe')
model.wv.most_similar(positive=['woman', 'king'], negative=['man'] ,topn=1 )
#computing the word2v for the documents

def document_vector(word2vec_model, doc):

    doc = [word for word in doc if word in word2vec_model.wv.vocab]

    return np.mean(word2vec_model[doc], axis=0)
def has_vector_representation(w2v_model, doc):

    """check if at least one word of the document is in the

    word2vec dictionary"""

    return not all(word not in w2v_model.wv.vocab for word in doc)
data =[]

i=[]

for index,doc in enumerate(text):#look up each doc in model

    if has_vector_representation(model,doc):

        data.append(document_vector(model, doc))

    else:

        i.append(index)
len(score),len(data)
score = np.delete(score,i)
len(score),len(data)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss

clf = SGDClassifier(loss = 'log',penalty = 'l2', n_jobs = -1)

#choose acc to best parameters

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

conf_log_ref = confusion_matrix(y_test, y_pred)



print("Confusion matrix:\n",conf_log_ref)

#ROC for a given hyperparameters for logistic regression

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class

probs = clf.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



#Plot ROC

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#print the log loss

a = log_loss(y_test, probs)

print("The log loss for the Logistic regression is:",a)



#print confusion matrix

show_confusion_matrix(conf_log_ref,['Negative','Positive'])



#Precision and recall

tn = conf_log_ref[0,0]; fp = conf_log_ref[0,1]; fn = conf_log_ref[1,0]; tp = conf_log_ref[1,1];



precision = 100*float(tp)/(tp+fp)

recall = 100*float(tp)/(tp+fn)



print("Precision :",precision)

print("Recall :",recall)



tp = conf_log_ref[0][0]

tn = conf_log_ref[1][1]

print("The accuracy is {} %".format(round(100.0*(tp+tn)/len(y_test),2)))