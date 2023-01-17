import re

import os

import numpy as np 

import pandas as pd 

from sklearn.svm import SVC

import plotly.express as px

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.metrics import recall_score,precision_score,make_scorer

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from imblearn.ensemble import BalancedRandomForestClassifier,EasyEnsembleClassifier,RUSBoostClassifier



pd.set_option('display.max_rows', 700)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_json('/kaggle/input/outofscope-intent-classification-dataset/is_train.json')

val = pd.read_json('/kaggle/input/outofscope-intent-classification-dataset/is_val.json')

test = pd.read_json('/kaggle/input/outofscope-intent-classification-dataset/is_test.json')

oos_train = pd.read_json('/kaggle/input/outofscope-intent-classification-dataset/oos_train.json')

oos_val = pd.read_json('/kaggle/input/outofscope-intent-classification-dataset/oos_val.json')

oos_test = pd.read_json('/kaggle/input/outofscope-intent-classification-dataset/oos_test.json')

files = [(train,'train'),(val,'val'),(test,'test'),(oos_train,'oos_train'),(oos_val,'oos_val'),(oos_test,'oos_test')]

for file,name in files:

    file.columns = ['text','intent']

    print(f'{name} shape:{file.shape}, {name} has {train.isna().sum().sum()} null values')

in_train = train.copy()
in_train.intent.value_counts()
def ngrams_top(corpus,ngram_range,n=10):

    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    total_list=words_freq[:n]

    df=pd.DataFrame(total_list,columns=['text','count'])

    return df



def get_emails(x):

    email = re.findall(r'[\w\.-]+@[\w-]+\.[\w]+',str(x))

    return " ".join(email)

def get_urls(x):

    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.[\w]+',str(x))

    return " ".join(url)

def get_mentions(x):

    mention = re.findall(r'(?<=@)\w+',str(x))

    return " ".join(mention)

def get_hashtags(x):

    hashtag = re.findall(r'(?<=#)\w+',str(x))

    return " ".join(hashtag)

def text_at_a_glance(df):

    res = df.apply(get_emails)

    res = res[res.values!=""]

    print("Data has {} rows with emails".format(len(res)))

    res = df.apply(get_urls)

    res = res[res.values!=""]

    print("Data has {} rows with urls".format(len(res)))

    res = df.apply(get_mentions)

    res = res[res.values!=""]

    print("Data has {} rows with mentions".format(len(res)))

    res = df.apply(get_hashtags)

    res = res[res.values!=""]

    print("Data has {} rows with hashtags".format(len(res)))
text_at_a_glance(in_train.text)
#check where hashtag has been found, punctuations will be taken care of by vectorizer()

temp = in_train.text.apply(get_hashtags)

in_train.iloc[temp[temp.values!=""].index].text
#top bigrams

bigrams = ngrams_top(in_train.text,(2,2)).sort_values(by='count')

px.bar(data_frame=bigrams,y='text',x='count',orientation='h',color_discrete_sequence=['#dc3912'],opacity=0.8,width=900,height=500)
#top trigrams

trigrams = ngrams_top(in_train.text,(3,3)).sort_values(by='count')

px.bar(data_frame=trigrams,y='text',x='count',orientation='h',color_discrete_sequence=['#f58518'],opacity=0.8,width=900,height=500)
def binarize(df):

    df.intent = np.where(df.intent!='oos',0,1)

    return df



def vectorizer(X):

    cv = CountVectorizer(min_df=1,ngram_range=(1,2))

    X_en = cv.fit_transform(X)

    return cv,X_en



def labelencoder(y):

    le = LabelEncoder()

    le.fit(y)

    y_enc = le.transform(y)

    return le,y_enc



def preprocess(train):

    X = train.text

    y = train.intent

    le,y = labelencoder(y)

    cv,X = vectorizer(X)

    return X,y,cv,le



def process_non_train(df,cv,le):

    X = df.text

    y = df.intent

    X = cv.transform(X)

    y = le.transform(y)

    return X,y



def get_score(clf,binary=0):

    clf.fit(X_train,y_train)

    if binary==1:

        y_pred = clf.predict(X_test)

        return clf,clf.score(X_val,y_val),clf.score(X_test,y_test),recall_score(y_test,y_pred),precision_score(y_test,y_pred)

    elif binary==0:

        return clf,clf.score(X_val,y_val),clf.score(X_test,y_test)
X_train,y_train,cv,le = preprocess(in_train)

X_val,y_val = process_non_train(val,cv,le)

X_test,y_test = process_non_train(test,cv,le)
val_scores = []

test_scores = []

names = []



models = [(KNeighborsClassifier(n_neighbors=15),'KNN'),(SGDClassifier(),'SGD clf'),(MultinomialNB(),'MultinomialNB'),

          (RandomForestClassifier(),'Random Forest'),(SVC(kernel='linear'),'Linear SVC')]



for model,name in models:

    clf,score,test_score = get_score(model,0)

    names.append(name)

    val_scores.append(score*100)

    test_scores.append(test_score*100)

pd.DataFrame(data=zip(val_scores,test_scores),index=names,columns=['val_score','test_score']).style.background_gradient()
params = {

    'loss':['squared_hinge','modified_huber'],

    'alpha':[0.0001,0.001,0.01],

    'max_iter':[250,500,1000],

    'validation_fraction':[0.2]

}

cv = GridSearchCV(SGDClassifier(random_state=111),param_grid=params, cv=5,n_jobs=-1,verbose=2)
cv.fit(X_train,y_train)
cv.best_params_
cv.best_score_
oos_plus_train = binarize(pd.concat([in_train,oos_train],axis=0).reset_index(drop=True))

oos_plus_val = binarize(pd.concat([val,oos_val],axis=0).reset_index(drop=True))

oos_plus_test = binarize(pd.concat([test,oos_test],axis=0).reset_index(drop=True))
oos_count = oos_plus_train.intent.value_counts()

oos_count
oos_count.plot(kind='bar')
X_train,y_train,cv,le = preprocess(oos_plus_train)

X_val,y_val = process_non_train(oos_plus_val,cv,le)

X_test,y_test = process_non_train(oos_plus_test,cv,le)
val_scores = []

test_scores = []

recall = []

names = []

precision = []



models = [(BalancedRandomForestClassifier(sampling_strategy='not minority',random_state=111),'Balanced Random Forest'),

          (RUSBoostClassifier(base_estimator=LogisticRegression(),sampling_strategy='not minority',random_state=111),'Random Undersampling + Adaboost'),

          (EasyEnsembleClassifier(n_estimators=30,base_estimator=LogisticRegression(),replacement=True,sampling_strategy='not minority',random_state=111),'Easy Ensemble')]



for model,name in models:

    _,score,test_score,recall_sc,precision_sc = get_score(model,1)

    names.append(name)

    val_scores.append(score*100)

    test_scores.append(test_score*100)

    recall.append(recall_sc*100)

    precision.append(precision_sc*100)

pd.DataFrame(data=zip(val_scores,test_scores,recall,precision),index=names,

             columns=['val_score','test_score','recall_score','precision_score']).style.background_gradient()
params = {

    'n_estimators':[30,50,70],

    'base_estimator':[AdaBoostClassifier(),LogisticRegression()],

    'replacement':[False,True]

}

cv = GridSearchCV(EasyEnsembleClassifier(n_jobs=-1,sampling_strategy='not minority',random_state=111),

                  verbose=2,

                  scoring='recall',

                  param_grid=params, cv=5,

                  n_jobs=-1)

cv.fit(X_train,y_train)
cv.best_score_
cv.best_params_
n_estimator = [25,50,100]

learning_rate=[0.001,0.0001]

result = []

for nest in n_estimator:

    for lr in learning_rate:

        eec = EasyEnsembleClassifier(base_estimator=AdaBoostClassifier(n_estimators=nest,learning_rate=lr)

                                     ,n_estimators=100,

                                     replacement=True,n_jobs=-1,

                                     sampling_strategy='not minority',random_state=111)

        eec.fit(X_train,y_train)

        recall = recall_score(y_test,eec.predict(X_test))

        result.append([nest,lr,recall])
pd.DataFrame(result,columns=['n_estimators','learning_rate','recall_score']).style.background_gradient(subset=['recall_score'])