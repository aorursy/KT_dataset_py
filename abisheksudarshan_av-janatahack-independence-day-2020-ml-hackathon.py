import pandas as pd

import numpy as np

import xgboost as xgb

import gc

from tqdm import tqdm

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from sklearn.metrics import classification_report,f1_score

stop_words = stopwords.words('english')

import seaborn as sns

import matplotlib.pyplot as plt

import string

import unidecode

import re

from skmultilearn.problem_transform import LabelPowerset# initialize label powerset multi-label classifier

%matplotlib inline

train=pd.read_csv('../input/janatahack-independence-day-2020-ml-hackathon/train.csv')

test=pd.read_csv('../input/janatahack-independence-day-2020-ml-hackathon/test.csv')
train.head(2)
test.head(2)
print('Train Shape: ',train.shape)

print('Test Shape: ',test.shape)
train.info()
test.info()
print('As count:\n')

print('Computer Science: ',train['Computer Science'].sum())

print('Physics: ',train['Physics'].sum())

print('Mathematics: ',train['Mathematics'].sum())

print('Statistics: ',train['Statistics'].sum())

print('Quantitative Biology: ',train['Quantitative Biology'].sum())

print('Quantiative Finance: ',train['Quantitative Finance'].sum())



print('\nAs a percentage:\n')

print('Computer Science: ',round(train['Computer Science'].sum()/train.shape[0]*100))

print('Physics: ',round(train['Physics'].sum()/train.shape[0]*100))

print('Mathematics: ',round(train['Mathematics'].sum()/train.shape[0]*100))

print('Statistics: ',round(train['Statistics'].sum()/train.shape[0]*100))

print('Quantitative Biology: ',round(train['Quantitative Biology'].sum()/train.shape[0]*100))

print('Quantiative Finance: ',round(train['Quantitative Finance'].sum()/train.shape[0]*100))

train['TITLE_len']=train['TITLE'].apply(len) 

test['TITLE_len']=test['TITLE'].apply(len) 



train['ABSTRACT_len']=train['ABSTRACT'].apply(len) 

test['ABSTRACT_len']=test['ABSTRACT'].apply(len) 



train['cons']=train['TITLE']+train['ABSTRACT'] 

test['cons']=test['TITLE']+test['ABSTRACT'] 



train['cons_len']=train['cons'].apply(len) 

test['cons_len']=test['cons'].apply(len) 

sns.distplot(train['TITLE_len'])

sns.distplot(test['TITLE_len'])
sns.distplot(train['ABSTRACT_len'])

sns.distplot(test['ABSTRACT_len'])
sns.distplot(train['cons_len'])

sns.distplot(test['cons_len'])
def remove_accented_chars(text):

    """remove accented characters from text, e.g. café"""

    text = unidecode.unidecode(text)

    return text
def lower_(text):

    return text.lower()
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

  

def stop_words_removal(sentence):

  

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(sentence)

  

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    return (' '.join(filtered_sentence))
stemmer = SnowballStemmer("english")



def stemming(sentence):

    

    stemSentence = ""

    for word in sentence.split():

        stem = stemmer.stem(word)

        stemSentence += stem

        stemSentence += " "

    stemSentence = stemSentence.strip()

    return stemSentence

def remove_special_characters(text, remove_digits=False):

    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'

    text = re.sub(pattern, '', text)

    return text
#Removing Ascents

train['TITLE']=train['TITLE'].apply(remove_accented_chars)

test['TITLE']=test['TITLE'].apply(remove_accented_chars)



train['ABSTRACT']=train['ABSTRACT'].apply(remove_accented_chars)

test['ABSTRACT']=test['ABSTRACT'].apply(remove_accented_chars)



train['cons']=train['cons'].apply(remove_accented_chars)

test['cons']=test['cons'].apply(remove_accented_chars)
train.head(2)
#Lower Casing the text

train['TITLE']=train['TITLE'].apply(lower_)

test['TITLE']=test['TITLE'].apply(lower_)



train['ABSTRACT']=train['ABSTRACT'].apply(lower_)

test['ABSTRACT']=test['ABSTRACT'].apply(lower_)



train['cons']=train['cons'].apply(lower_)

test['cons']=test['cons'].apply(lower_)
train.head(2)
#Removing Special Characters

train['TITLE']=train['TITLE'].apply(remove_special_characters)

test['TITLE']=test['TITLE'].apply(remove_special_characters)



train['ABSTRACT']=train['ABSTRACT'].apply(remove_special_characters)

test['ABSTRACT']=test['ABSTRACT'].apply(remove_special_characters)



train['cons']=train['cons'].apply(remove_special_characters)

test['cons']=test['cons'].apply(remove_special_characters)
train.head(2)
#Stopwords removal

train['TITLE']=train['TITLE'].apply(stop_words_removal)

test['TITLE']=test['TITLE'].apply(stop_words_removal)



train['ABSTRACT']=train['ABSTRACT'].apply(stop_words_removal)

test['ABSTRACT']=test['ABSTRACT'].apply(stop_words_removal)



train['cons']=train['cons'].apply(stop_words_removal)

test['cons']=test['cons'].apply(stop_words_removal)
train.head(2)
#Stemming

train['TITLE']=train['TITLE'].apply(stemming)

test['TITLE']=test['TITLE'].apply(stemming)



train['ABSTRACT']=train['ABSTRACT'].apply(stemming)

test['ABSTRACT']=test['ABSTRACT'].apply(stemming)



train['cons']=train['cons'].apply(stemming)

test['cons']=test['cons'].apply(stemming)
train.head(2)
#Writing the pre-processed text data

train.to_csv('new_train.csv')

test.to_csv('new_test.csv')
train=pd.read_csv('../input/preprocessed-av-topic-modelling/new_train.csv')

test=pd.read_csv('../input/preprocessed-av-topic-modelling/new_test.csv')
train['title_orig_len']=train['TITLE_len']

test['title_orig_len']=test['TITLE_len']



train['abs_orig_len']=train['ABSTRACT_len']

test['abs_orig_len']=test['ABSTRACT_len']



train['cons_orig_len']=train['cons_len']

test['cons_orig_len']=test['cons_len']
train['TITLE_len']=train['TITLE'].apply(len) 

test['TITLE_len']=test['TITLE'].apply(len) 



train['ABSTRACT_len']=train['ABSTRACT'].apply(len) 

test['ABSTRACT_len']=test['ABSTRACT'].apply(len) 



train['cons_len']=train['cons'].apply(len) 

test['cons_len']=test['cons'].apply(len) 
train.describe()
test.describe()
sns.heatmap(train.drop(['ID','title_orig_len','abs_orig_len','cons_orig_len','TITLE_len','ABSTRACT_len'],axis=1).corr(),annot=True)
tr,ev = train_test_split(train,random_state=101,test_size=0.3, shuffle=True)
# Always start with these features. They work (almost) everytime!

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(list(train['cons'].values) + list(test['cons'].values))



#Train

xtrain_tfv =  tfv.transform(tr['cons']) 

xvalid_tfv = tfv.transform(ev['cons'])



#Test

xtest_tfv= tfv.transform(test['cons'])
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), stop_words = 'english')



# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)

ctv.fit(list(train['cons'].values) + list(test['cons'].values))



#Train

xtrain_ctv =  ctv.transform(tr['cons']) 

xvalid_ctv = ctv.transform(ev['cons'])



#Test

xtest_ctv= ctv.transform(test['cons'])
# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.

svd = decomposition.TruncatedSVD(n_components=120)

svd.fit(xtrain_tfv)

xtrain_svd = svd.transform(xtrain_tfv)

xvalid_svd = svd.transform(xvalid_tfv)

xtest_svd = svd.transform(xtest_tfv)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = preprocessing.StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)

xtest_svd_scl = scl.transform(xtest_svd)
#targets that need to be predicted

targets=['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']



#Empty data frame for predictions

ev_pred=pd.DataFrame()

test_pred=pd.DataFrame()
#Using LogisticRegression one at a time on tf_idf

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using LogisticRegression

    classifier = LogisticRegression()

    classifier.fit(xtrain_tfv, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_tfv)

    test_pred[t] = classifier.predict(xtest_tfv)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_logit.csv', index=False)

print("Your submission was successfully saved!")
#Using LogisticRegression one at a time on ctv

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using LogisticRegression

    classifier = LogisticRegression(max_iter=10000)

    classifier.fit(xtrain_ctv, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_ctv)

    test_pred[t] = classifier.predict(xtest_ctv)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('ctv_logit.csv', index=False)

print("Your submission was successfully saved!")
#Using MultinomialNB one at a time on tf_idf

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using MultinomialNB

    classifier = MultinomialNB()

    classifier.fit(xtrain_tfv, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_tfv)

    test_pred[t] = classifier.predict(xtest_tfv)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_mnb.csv', index=False)

print("Your submission was successfully saved!")
#Using MultinomialNB one at a time on ctv

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using MultinomialNB

    classifier = MultinomialNB()

    classifier.fit(xtrain_ctv, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_ctv)

    test_pred[t] = classifier.predict(xtest_ctv)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('ctv_mnb.csv', index=False)

print("Your submission was successfully saved!")
#Using SVC One at a time

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using XGB

    classifier = SVC() 

    classifier.fit(xtrain_svd_scl, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_svd_scl)

    test_pred[t] = classifier.predict(xtest_svd_scl)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_svc.csv', index=False)

print("Your submission was successfully saved!")
#Using XGBoost one at a time

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using XGB

    classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

    classifier.fit(xtrain_tfv, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_tfv)

    test_pred[t] = classifier.predict(xtest_tfv)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_xgb.csv', index=False)

print("Your submission was successfully saved!")
#Using XGBClassifier one at a time on ctv

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using XGBClassifier

    classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

    classifier.fit(xtrain_ctv, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_ctv)

    test_pred[t] = classifier.predict(xtest_ctv)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('ctv_xgb.csv', index=False)

print("Your submission was successfully saved!")
#Using XGBoost on SVD components

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using XGB789746001881468

    classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

    classifier.fit(xtrain_svd, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_svd)

    test_pred[t] = classifier.predict(xtest_svd)



for t in targets:

    print(t)

    print(f1_score(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_svd_xgb.csv', index=False)

print("Your submission was successfully saved!")
#Using XGBoost on SVD components

for t in targets:



    y_train=tr[t]

    y_test=ev[t]



    #using XGB

    classifier = xgb.XGBClassifier(nthread=10)

    classifier.fit(xtrain_svd, y_train)

    

    ev_pred[t] = classifier.predict(xvalid_svd)

    test_pred[t] = classifier.predict(xtest_svd)



for t in targets:

    print(t)

    print(classification_report(ev[t],ev_pred[t]))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_svd_xgb2.csv', index=False)

print("Your submission was successfully saved!")
# Initialize SVD

svd = TruncatedSVD()

    

# Initialize the standard scaler 

scl = preprocessing.StandardScaler()



# We will use logistic regression here..

lr_model = LogisticRegression()



# Create the pipeline 

clf = pipeline.Pipeline([('svd', svd),

                         ('scl', scl),

                         ('lr', lr_model)])
# Next we need a grid of parameters:



param_grid = {'svd__n_components' : [120, 180],

              'lr__C': [0.1, 1.0, 10], 

              'lr__penalty': ['l1', 'l2']}
for t in targets:



    y_train=tr[t]

    

    print('\n For',t)

    # Initialize Grid Search Model

    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1_micro',

                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)



    # Fit Grid Search Model

    model.fit(xtrain_tfv, y_train)  # we can use the full data here but im only using xtrain

    print("Best score: %0.3f" % model.best_score_)

    print("Best parameters set:")

    best_parameters = model.best_estimator_.get_params()

    for param_name in sorted(param_grid.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    

'''

    Results from GridSerach CV

    

    For Computer Science:

    Best score: 0.857

    Best parameters set:

        lr__C: 0.1

        lr__penalty: 'l2'

        svd__n_components: 180



    For Physics:

    Best score: 0.932

    Best parameters set:

        lr__C: 1.0

        lr__penalty: 'l2'

        svd__n_components: 120

    

    For Mathematics:

    Best score: 0.902

    Best parameters set:

        lr__C: 0.1

        lr__penalty: 'l2'

        svd__n_components: 120

    

    For Statistics:

    Best score: 0.881

    Best parameters set:

        lr__C: 1.0

        lr__penalty: 'l2'

        svd__n_components: 180

    

    For Quantitative Biology:

    Best score: 0.974

    Best parameters set:

        lr__C: 10

        lr__penalty: 'l2'

        svd__n_components: 180



    For Quantitative Finance:

    Best score: 0.990

    Best parameters set:

        lr__C: 1.0

        lr__penalty: 'l2'

        svd__n_components: 120



'''
# Using Grid Search Results:



svd_comp={'Computer Science': 180, 'Physics': 120, 'Mathematics': 120, 'Statistics': 180, 'Quantitative Biology': 180, 'Quantitative Finance':120}

lr_c={'Computer Science': 0.1, 'Physics': 1.0, 'Mathematics': 0.1, 'Statistics': 1, 'Quantitative Biology': 10, 'Quantitative Finance':1}

lr_pen={'Computer Science': 'l2', 'Physics': 'l2', 'Mathematics': 'l2', 'Statistics': 'l2', 'Quantitative Biology': 'l2', 'Quantitative Finance':'l2'}



for t in targets:

    

    y_train=tr[t]

    

    # Initialize SVD

    svd = TruncatedSVD(n_components=svd_comp[t])

    

    # Initialize the standard scaler 

    scl = preprocessing.StandardScaler()



    # We will use logistic regression here..

    lr_model = LogisticRegression(C=lr_c[t],penalty=lr_pen[t])

    

    svd.fit(xtrain_tfv)

    xtrain_svd = svd.transform(xtrain_tfv)

    xvalid_svd = svd.transform(xvalid_tfv)

    xtest_svd = svd.transform(xtest_tfv)



    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.

    scl.fit(xtrain_svd)

    xtrain_svd_scl = scl.transform(xtrain_svd)

    xvalid_svd_scl = scl.transform(xvalid_svd)

    xtest_svd_scl = scl.transform(xtest_svd)

    

    # Model Fit

    lr_model.fit(xtrain_svd_scl, y_train)  

    

    ev_pred[t] = lr_model.predict(xvalid_svd_scl)

    test_pred[t] = lr_model.predict(xtest_svd_scl)



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_svd_logit_gsv.csv', index=False)

print("Your submission was successfully saved!")
# Lightgbm

import lightgbm as lgb
for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]

    

    clf = lgb.LGBMClassifier(n_estimators=450,learning_rate=0.03,random_state=42,colsample_bytree=0.5,reg_alpha=2,reg_lambda=2)

    

    clf.fit(xtrain_svd_scl, y_train, early_stopping_rounds=100, eval_set=[(xtrain_svd_scl, y_train), (xvalid_svd_scl, y_test)], eval_metric='f1_micro', verbose=True)



    eval_score = f1_score(y_test, clf.predict(xvalid_svd_scl))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = clf.predict(xvalid_svd_scl)

    test_pred[t] = clf.predict(xtest_svd_scl)



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('tf_idf_svd_lgbm.csv', index=False)

print("Your submission was successfully saved!")
embeddings_index = {}

f = open('../input/glove840b300dtxt/glove.840B.300d.txt')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
# this function creates a normalized vector for the whole sentence

def sent2vec(s):

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        try:

            M.append(embeddings_index[w])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set

xtrain_glove = [sent2vec(x) for x in tqdm(tr['cons'])]

xvalid_glove = [sent2vec(x) for x in tqdm(ev['cons'])]

xtest_glove = [sent2vec(x) for x in tqdm(test['cons'])]

# Fitting a simple xgboost on glove features

for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]



    clf = xgb.XGBClassifier(nthread=10, silent=False)

    clf.fit(np.array(xtrain_glove), y_train)

    eval_score = f1_score(y_test, clf.predict(np.array(xvalid_glove)))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = clf.predict(np.array(xvalid_glove))

    test_pred[t] = clf.predict(np.array(xtest_glove))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('glove_xgb.csv', index=False)

print("Your submission was successfully saved!")

# Fitting a simple xgboost on glove features

for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]



    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)

    clf.fit(np.array(xtrain_glove), y_train)

    eval_score = f1_score(y_test, clf.predict(np.array(xvalid_glove)))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = clf.predict(np.array(xvalid_glove))

    test_pred[t] = clf.predict(np.array(xtest_glove))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('glove_xgb2.csv', index=False)

print("Your submission was successfully saved!")

# scale the data before any neural net:

scl = preprocessing.StandardScaler()

xtrain_glove_scl = scl.fit_transform(xtrain_glove)

xvalid_glove_scl = scl.transform(xvalid_glove)

xtest_glove_scl = scl.transform(xtest_glove)
# create a simple 3 layer sequential neural net

model = Sequential()



model.add(Dense(300, input_dim=300, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(150, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(30, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(units=1,activation='softmax'))



# compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]



    

    model.fit(x=xtrain_glove_scl,

              y=y_train, 

              batch_size=256, 

              epochs=500, 

              verbose=1, 

              validation_data=(xvalid_glove_scl, y_test),

              callbacks=[early_stop]

             )

        

    eval_score = f1_score(y_test, model.predict(xvalid_glove_scl).astype('int'))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = np.array(model.predict(xvalid_glove_scl))

    test_pred[t] = np.array(model.predict(xtest_glove_scl))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('glove_nn.csv', index=False)

print("Your submission was successfully saved!")

# using keras tokenizer here

token = text.Tokenizer(num_words=None)

max_len = 70



token.fit_on_texts(list(train['cons']) + list(test['cons']))

xtrain_seq = token.texts_to_sequences(tr['cons'])

xvalid_seq = token.texts_to_sequences(ev['cons'])

xtest_seq = token.texts_to_sequences(test['cons'])





# zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)



word_index = token.word_index
# create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

# A simple LSTM with glove embeddings and two dense layers

model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     300,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]



    

    model.fit(xtrain_pad, y=y_train, batch_size=512, epochs=100, 

          verbose=1, validation_data=(xvalid_pad, y_test), callbacks=[earlystop])



        

    eval_score = f1_score(y_test, model.predict(xvalid_pad).astype('int'))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = np.array(model.predict(xvalid_pad))

    test_pred[t] = np.array(model.predict(xtest_pad))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('lstm_nn.csv', index=False)

print("Your submission was successfully saved!")
xtrain_pad.shape
# A simple bidirectional LSTM with glove embeddings and two dense layers



model = Sequential()

model.add(Embedding(len(word_index) + 1,

                         300,

                         weights=[embedding_matrix],

                         input_length=max_len,

                         trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]



    

    model.fit(xtrain_pad, y=y_train, batch_size=512, epochs=100,verbose=1, validation_data=(xvalid_pad, y_test), callbacks=[earlystop])

    

    eval_score = f1_score(y_test, model.predict(xvalid_pad).astype('int'))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = np.array(model.predict(xvalid_pad))

    test_pred[t] = np.array(model.predict(xtest_pad))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('bilstm_nn.csv', index=False)

print("Your submission was successfully saved!")



# GRU with glove embeddings and two dense layers

model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     300,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))

model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

for t in targets:

    

    y_train=tr[t]

    y_test=ev[t]



    

    model.fit(xtrain_pad, y=y_train, batch_size=512, epochs=100, 

          verbose=1, validation_data=(xvalid_pad, y_test), callbacks=[earlystop])

    

    eval_score = f1_score(y_test, model.predict(xvalid_pad).astype('int'))

    

    print('Eval ACC: {}'.format(eval_score))

    

    ev_pred[t] = np.array(model.predict(xvalid_pad))

    test_pred[t] = np.array(model.predict(xtest_pad))



output = pd.DataFrame({'ID': test['ID'], 'Computer Science':test_pred['Computer Science'],'Physics':test_pred['Physics'],'Mathematics':test_pred['Mathematics'],'Statistics':test_pred['Statistics'],'Quantitative Biology':test_pred['Quantitative Biology'],'Quantitative Finance':test_pred['Quantitative Finance'] })

output.to_csv('gru_nn.csv', index=False)

print("Your submission was successfully saved!")


