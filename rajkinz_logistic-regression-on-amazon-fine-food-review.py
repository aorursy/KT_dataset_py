import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
df = pd.read_csv("../input/amazon-fine-food-corpus/corpus.csv")
df.head()
df1k = df.loc[:119999,:]
print("Shape:- ",df1k.shape)
print(df1k.head())
df1k['Score'].value_counts()
from sklearn.model_selection import TimeSeriesSplit
def timesplit(x,y):
    ts = TimeSeriesSplit(n_splits = 4)
    for train_index,test_index in ts.split(x):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = timesplit(df1k["Text"].values,df1k["Score"].values)
def imp_features(model,classifier):
    voc = model.get_feature_names()
    w = list(classifier.coef_[0])
    pos_coef = []
    neg_coef = []
    pos_words = []
    neg_words = []
    for i,c in enumerate(w):
        if c > 0:
            pos_coef.append(c)
            pos_words.append(voc[i])
        if c < 0:
            neg_coef.append(abs(c))
            neg_words.append(voc[i])
    pos_df = pd.DataFrame(columns = ['Words','Coef'])
    neg_df = pd.DataFrame(columns = ['Words','Coef'])
    pos_df['Words'] = pos_words
    pos_df['Coef'] = pos_coef
    neg_df['Words'] = neg_words
    neg_df['Coef'] = neg_coef
    pos_df = pos_df.sort_values("Coef",axis = 0,ascending = False).reset_index(drop=True)
    neg_df = neg_df.sort_values("Coef",axis = 0,ascending = False).reset_index(drop=True)
    print("Shape of Positive dataframe:- ,",pos_df.shape)
    print("Shape of Negative dataframe:- ",neg_df.shape)
    print("Top ten positive predictors:- \n",pos_df.head(10))
    print("\nTop ten negative predictors:- \n",neg_df.head(10))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bow_train = cv.fit_transform(x_train)
print("Shape of BOW vector:- ",bow_train.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
bow_train = sc.fit_transform(bow_train)
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty = 'l2',solver = 'sag')
param_grid = {"C":[0.01,0.1,1,5,10,50]}
gs = GridSearchCV(classifier,param_grid,cv = 5,scoring = 'f1_micro',n_jobs = -1)
gs.fit(bow_train,y_train)
print("Best parameter:- ",gs.best_params_)
print("Best score:- ",gs.best_score_)
%%time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
bow_test = cv.transform(x_test)
bow_test = sc.transform(bow_test)   # Standardizing the test data
classifier = LogisticRegression(C=1,penalty = 'l2',solver = 'sag',n_jobs = -1)
classifier.fit(bow_train,y_train)
y_pred = classifier.predict(bow_test)
print("BOW test accuracy:- ",accuracy_score(y_test,y_pred))
print("F1 score:- ",f1_score(y_test,y_pred,average='micro'))
print("Training accuracy:- ",accuracy_score(y_train,classifier.predict(bow_train)))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)
%%time
from scipy.sparse import csr_matrix
coef = classifier.coef_    #weight vector of original classifier
e = 0.02 # introducing small error in the training dataset
bow_train_pert = csr_matrix(bow_train,dtype=np.float64)
bow_train_pert[np.nonzero(bow_train_pert)]+=e
classifier_pert = LogisticRegression(C=1,penalty = 'l2',solver = 'sag',n_jobs = -1)
classifier_pert.fit(bow_train_pert,y_train)
coef_pert = classifier_pert.coef_
coef_diff = coef_pert - coef
print("Average difference in weight vectors:- ",np.mean(coef_diff))
imp_features(cv,classifier)
y_pred_prob = classifier.predict_proba(bow_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True)
tfidf_train = tfidf.fit_transform(x_train)
print("Shape of tfidf_train:- ",tfidf_train.shape)
sc = StandardScaler(with_mean = False)
tfidf_train = sc.fit_transform(tfidf_train)
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty = 'l1',solver = 'liblinear',class_weight = 'balanced')
param_grid = {"C":[0.01,0.1,1,10,50]}
gs = GridSearchCV(classifier,param_grid,cv = 5,scoring = 'f1',n_jobs = -1)
gs.fit(tfidf_train,y_train)
print("Best parameter:- ",gs.best_params_)
print("Best score:- ",gs.best_score_)
%%time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
tfidf_test = tfidf.transform(x_test)
tfidf_test = sc.transform(tfidf_test)   # Standardizing the test data
classifier = LogisticRegression(C=0.01,penalty = 'l1',solver = 'liblinear',class_weight = 'balanced')
classifier.fit(tfidf_train,y_train)
y_pred = classifier.predict(tfidf_test)
print("Tfdif test accuracy:- ",accuracy_score(y_test,y_pred))
print("F1 score:- ",f1_score(y_test,y_pred))
print("Training accuracy:- ",accuracy_score(y_train,classifier.predict(tfidf_train)))
%%time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
tfidf_test = tfidf.transform(x_test)
tfidf_test = sc.transform(tfidf_test)   # Standardizing the test data
classifier = LogisticRegression(C=0.01,penalty = 'l1',solver = 'liblinear')
classifier.fit(tfidf_train,y_train)
y_pred = classifier.predict(tfidf_test)
print("Tfdif test accuracy:- ",accuracy_score(y_test,y_pred))
print("F1 score:- ",f1_score(y_test,y_pred))
print("Training accuracy:- ",accuracy_score(y_train,classifier.predict(tfidf_train)))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)
print("No. of features with zero coefficients:- ",tfidf_train.shape[1]-np.count_nonzero(classifier.coef_))
imp_features(tfidf,classifier)
y_pred_prob = classifier.predict_proba(tfidf_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()
y_pred_prob = classifier.predict_proba(tfidf_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()
#Function to create list of sentences
def sent_list(x):
    list_of_sent = []
    for sent in tqdm(x):
        words = []
        for w in sent.split():
            words.append(w)
        list_of_sent.append(words)
    return list_of_sent
#implementing word2vec
from gensim.models import Word2Vec
sent_train = sent_list(x_train)
w2v = Word2Vec(sent_train,size=50,min_count=2,workers=4)
#Function to create avg word2vec vector
def avgw2v(x):
    avgw2v_vec = []
    for sent in tqdm(x):
        sent_vec = np.zeros(50)
        count = 0
        for word in sent:
            try:
                vec = w2v.wv[word]
                sent_vec+=vec
                count+=1
            except:
                pass
        sent_vec/=count
        avgw2v_vec.append(sent_vec)
    return avgw2v_vec
#Creating average word2vec training data
avgw2v_train = np.array(avgw2v(sent_train))
print("Shape of avg word2vec train data:- ",avgw2v_train.shape)
sc = StandardScaler()
avgw2v_train = sc.fit_transform(avgw2v_train)
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty = 'l2',solver='sag',class_weight = 'balanced')
param_grid = {"C":[0.01,0.1,1,10,50]}
gs = GridSearchCV(classifier,param_grid,cv = 5,scoring = 'f1',n_jobs = -1)
gs.fit(avgw2v_train,y_train)
print("Best parameter:- ",gs.best_params_)
print("Best score:- ",gs.best_score_)
%%time
sent_test = sent_list(x_test)
avgw2v_test = np.array(avgw2v(sent_test))
avgw2v_test = sc.transform(avgw2v_test)
classifier = LogisticRegression(C=1,penalty = 'l2',solver = 'sag',class_weight = 'balanced')
classifier.fit(avgw2v_train,y_train)
y_pred = classifier.predict(avgw2v_test)
print("Avg Word2Vec test accuracy:- ",accuracy_score(y_test,y_pred))
print("F1 score:- ",f1_score(y_test,y_pred))
print("Training accuracy:- ",accuracy_score(y_train,classifier.predict(avgw2v_train)))
%%time
sent_test = sent_list(x_test)
avgw2v_test = np.array(avgw2v(sent_test))
avgw2v_test = sc.transform(avgw2v_test)
classifier = LogisticRegression(C=1,penalty = 'l2',solver = 'sag')
classifier.fit(avgw2v_train,y_train)
y_pred = classifier.predict(avgw2v_test)
print("Avg Word2Vec test accuracy:- ",accuracy_score(y_test,y_pred))
print("F1 score:- ",f1_score(y_test,y_pred))
print("Training accuracy:- ",accuracy_score(y_train,classifier.predict(avgw2v_train)))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)
y_pred_prob = classifier.predict_proba(avgw2v_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()
#Function for creating tfidf weighted Word2Vec
def tfidfw2v(x):
    dictionary = dict(zip(tfidf.get_feature_names(),list(tfidf.idf_)))
    tfidf_w2v_vec = []
    i=0
    for sent in tqdm(x):
        sent_vec = np.zeros(50)
        weights = 0
        for word in sent:
            try:
                vec = w2v.wv[word]
                tfidf_value = dictionary[word]*sent.count(word)
                sent_vec+=(tfidf_value*vec)
                weights+=tfidf_value
            except:
                pass
        sent_vec/=weights
        tfidf_w2v_vec.append(sent_vec)
        i+=1
    return tfidf_w2v_vec
tfidfw2v_train = np.array(tfidfw2v(sent_train))
print("Shape of tfidf avgw2v train vector:- ",tfidfw2v_train.shape)
sc = StandardScaler()
tfidfw2v_train = sc.fit_transform(tfidfw2v_train)
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty = 'l2',solver='sag',class_weight = 'balanced')
param_grid = {"C":[0.1,1,10,50,100]}
gs = GridSearchCV(classifier,param_grid,cv = 5,scoring = 'f1',n_jobs = -1)
gs.fit(tfidfw2v_train,y_train)
print("Best parameter:- ",gs.best_params_)
print("Best score:- ",gs.best_score_)
%%time
sent_test = sent_list(x_test)
tfidfw2v_test = np.array(tfidfw2v(sent_test))
tfidfw2v_test = sc.transform(tfidfw2v_test)
classifier = LogisticRegression(C=10,penalty = 'l2',solver = 'sag',class_weight = 'balanced')
classifier.fit(tfidfw2v_train,y_train)
y_pred = classifier.predict(tfidfw2v_test)
print("Tfidf Word2Vec test accuracy:- ",accuracy_score(y_test,y_pred))
print("F1 score:- ",f1_score(y_test,y_pred))
print("Training accuracy:- ",accuracy_score(y_train,classifier.predict(tfidfw2v_train)))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)
y_pred_prob = classifier.predict_proba(tfidfw2v_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()