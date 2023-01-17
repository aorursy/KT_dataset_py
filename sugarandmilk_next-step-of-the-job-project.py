# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import gc
import re
import xgboost as xgb

!pip install pymystem3

from nltk.corpus import stopwords
from pymystem3 import mystem
from string import punctuation

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
SEED = 42
def EncodeText(train, test):
    '''Encoding text data by TF-IDF'''
    
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.90,ngram_range=(1,2),max_features=10000)
    train = vectorizer.fit_transform(train)
    test = vectorizer.transform(test)
    return train, test, vectorizer
def TrainLinearSVC(X_train, y_train):
    '''Training LinearSVC with and without class balancing'''

    clf_svc = LinearSVC()   
    clf_svc.fit(X_train, y_train)
    return clf_svc

def TrainXgboost(X_train, X_test, y_train, y_test, n_classes, eta=0.05, n_est=100):
    '''Training the XGBoost classifier'''
    
    params = {'booster' : 'gbtree','objective' : 'multi:softmax', 
              'nthread': -1, 'num_class' : n_classes, 'eta' : eta, 
              'n_estimators' : n_est, 'seed' : 42 }
       
    X_train = scipy.sparse.csc_matrix(X_train)
    X_test = scipy.sparse.csc_matrix(X_test)
    
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)

    clf = xgb.train(params, xgb_train)
   
    f1_sc =  f1_score(y_test, clf.predict(xgb_test), average = 'micro')
    
    print('F1_Score :')
    print(f1_sc)
    
    return f1_sc
def GetRandForestClass(XTrain, yTrain):
    '''Training the Random Forest Classifier'''
    
    RF_model = RandomForestClassifier(n_jobs = -1, n_estimators = 150, random_state = SEED)
    RF_model.fit(XTrain, yTrain)
    return RF_model
def TrainLogReg(X_train, y_train):
    '''Training the Logistic Regression'''
    
    #params = {flag_balance : 'True', n_jobs : -1, solver : 'saga'}
    log_Reg_model = LogisticRegression(random_state = SEED, class_weight='balanced')
    log_Reg_model.fit(X_train, y_train)
    return log_Reg_model
def GetImportantNGramms(vectorize_tfidf, text, n_words):
    '''Get most weighted N-Gramms. This function requires a lot of RAM for full text.'''
    
    feature_array = np.array(vectorize_tfidf.get_feature_names())
    tfidf_sorting = np.argsort(text.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n_words]
    
    return top_n
def TrainNB(X_train, y_train):
    '''Training a Naive Bayesian Classifier with OneVsRest Technology'''
    
    clf_nb = MultinomialNB()
    clf = OneVsRestClassifier(clf_nb)
    clf.fit(X_train, y_train)
    return clf

def TrainSGDClass(X_train, y_train):
    '''Training the SGDClassifier  '''
    
    SGD_model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=SEED, max_iter=5, tol=None, class_weight='balanced')
    SGD_model.fit(X_train, y_train)
    return SGD_model
    
def GetScores(model, X_test, y_test):
    '''Getting metrics: f1, precision, recall for each class'''
    
    y_predict = model.predict(X_test)
    f1_sc =  f1_score(y_test, y_predict, average = 'micro')
    
    
    print('F1_Score :')
    print(f1_sc)
    print ('Report : ')
    print (classification_report(y_test, y_predict))
punctuation = list(punctuation)
punctuation += '«»—'
mystem = mystem.Mystem()

def preprocess_text(text):
    """Text preprocess function"""
    
    russian_stopwords = stopwords.words("russian")
    extra_stopwords = ['который', 'это', 'свой','также']
    russian_stopwords.extend(extra_stopwords)
    
    tokens = mystem.lemmatize(text.lower())
    
    tokens = [token for token in tokens
              if token not in russian_stopwords
              and token != " "
              and token.strip() not in punctuation
              and not token.isdigit()]

    text = " ".join(tokens)

    return text
def Topic_encoder(topic):
    '''Encodes a topic into numbers'''
    
    topic_encoder = LabelEncoder()
    topic = topic_encoder.fit_transform(topic)
    return topic
filename = "/kaggle/input/a-job-project/textFromEDA.csv" #The path to the file
df = pd.read_csv(filename)
df.head(5)
%%time
df['text'] = [preprocess_text(df['text'][i]) 
               for i in range(len(df['text']))]
%%time
df['title'] = [preprocess_text(df['title'][i]) 
               for i in range(len(df['title']))]
df.to_csv('preprocess_text.csv', index = False)
#filename = "/kaggle/input/a-job-project/preprocess_text2.csv" #The path to the file
#df = pd.read_csv(filename)
#df.head(5)
X = df['text'] + ' ' + df['title']
y = Topic_encoder(df['topic'])

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.30,
                                                    random_state=42, stratify=y)

del X, y
gc.collect()
%%time
XTrain, XTest, tfidf = EncodeText(XTrain, XTest)
stop
#GetImportantNGramms(tfidf,  XTest[:1000], 100)
%%time
XGBy_predict = TrainXgboost(XTrain, XTest, yTrain, yTest, len(np.unique(yTrain)))
XGBy_predict
%%time
SVC_model = TrainLinearSVC(XTrain, yTrain)
GetScores(SVC_model, XTest, yTest)
%%time
NB_model = TrainNB(XTrain, yTrain)
GetScores(NB_model, XTest, yTest)
%%time
LogReg_model = TrainLogReg(XTrain, yTrain)
GetScores(LogReg_model, XTest, yTest)
%%time
SGD_model = TrainSGDClass(XTrain, yTrain)
GetScores(SGD_model, XTest, yTest)
%%time
RandForestModel = GetRandForestClass(XTrain, yTrain)
GetScores(RandForestModel, XTest, yTest)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
#model1 = LogisticRegression()
#model2 = DecisionTreeClassifier()
#model3 = SVC()
estimators.append(('logistic', LogReg_model))
estimators.append(('nb', NB_model))
estimators.append(('svm', SVC_model))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, XTrain, yTrain, cv=5, scoring='f1_macro')
print(results.mean())
%%time
SVC_model = TrainLinearSVC(lda_train_matrix, yTrain)

GetScores(SVC_model, lda_test_matrix, yTest)
