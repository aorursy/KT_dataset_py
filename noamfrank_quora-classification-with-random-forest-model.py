################# GENERAL IMPORTS

import os

import string

from pprint import pprint

from operator import itemgetter

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score

##################################### NLP SPECIFIC IMPORTS

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk import ngrams

from nltk.stem import PorterStemmer

from nltk.corpus import reuters

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud

from collections import Counter

reuters.fileids()

stopwords.words('english')



import numpy as np

import pandas as pd

import warnings

from sys import modules



warnings.filterwarnings('ignore')

%matplotlib inline



from gensim.models import word2vec

import logging



from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten

from keras.layers.wrappers import TimeDistributed

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM

from keras.layers import Dropout



import seaborn as sns

sns.set(style = 'darkgrid')

print(os.listdir("../input"))

import re

pd.set_option('max_colwidth', 800)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



np.random.seed(1234)

%matplotlib inline



print('all set')
quora_train=pd.read_csv("../input/train.csv")

quora_test=pd.read_csv("../input/test.csv")

print("Train size =" ,quora_train.shape)

print("Test size =" ,quora_test.shape)
# quora_train=quora_train[0:100000]

# quora_test=quora_test[0:20000]
quora_train['words'] = quora_train.question_text.apply(lambda x: len(x.split()))

quora_train['characters'] = quora_train.question_text.apply(lambda x: len(x))

quora_test['words'] = quora_test.question_text.apply(lambda x: len(x.split()))

quora_test['characters'] = quora_test.question_text.apply(lambda x: len(x))
fig = plt.figure(figsize=(18, 7))



plt.subplot(1, 2, 1)

quora_train.groupby('target')['words'].mean().plot(kind='bar', ylim=(0,20), title= 'Average word count by target')



plt.subplot(1, 2, 2)

quora_train.groupby('target')['characters'].mean().plot(kind='bar', ylim=(0,105), title= 'Average character count by target')
from nltk import pos_tag



def verb_count(text):

    token_text= word_tokenize(text)

    tagged_text = pos_tag(token_text)

    counter=0

    for w,t in tagged_text:

        t = t[:2]

        if t in ['VB']:

            counter+=1

    return counter



def noun_count(text):

    token_text= word_tokenize(text)

    tagged_text = pos_tag(token_text)

    counter=0

    for w,t in tagged_text:

        t = t[:2]

        if t in ['NN']:

            counter+=1

    return counter

quora_train['question_text_prep'] = quora_train['question_text'].apply(lambda x: x.lower())

quora_test['question_text_prep'] = quora_test['question_text'].apply(lambda x: x.lower())
def pad_punctuation_w_space(string):

    s = re.sub('([:;"*.,!?()/\=-])', r' \1 ', string)

    s=re.sub('[^a-zA-Z]',' ',s)

    s = re.sub('\s{2,}', ' ', s)

    s =  re.sub(r"\b[a-zA-Z]\b", "", s) #code for removing single characters

    return s

quora_train['question_text_prep'] = quora_train['question_text_prep'].apply(lambda x: pad_punctuation_w_space(x))

quora_test['question_text_prep'] = quora_test['question_text_prep'].apply(lambda x: pad_punctuation_w_space(x))

quora_train['question_text_prep'] = quora_train['question_text_prep'].apply(lambda x: x.split())

quora_test['question_text_prep'] = quora_test['question_text_prep'].apply(lambda x: x.split())
stop_list = stopwords.words('english') + list(string.punctuation)

quora_train['question_text_prep'] = quora_train['question_text_prep'].apply(lambda x: [i for i in x if i not in stop_list])

quora_test['question_text_prep'] = quora_test['question_text_prep'].apply(lambda x: [i for i in x if i not in stop_list]) 
quora_train['question_text_prep_string'] = quora_train['question_text_prep'].str.join(" ")

quora_test['question_text_prep_string'] = quora_test['question_text_prep'].str.join(" ")
sents = list(quora_train.question_text_prep.values) 

sents[0]
min_num = 3 # minimum number of occurrences in text

EMBEDDING_FILE= "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
import numpy as np

def loadGloveModel(gloveFile):

    print ("Loading Glove Model")

    f = open(gloveFile,'r', encoding='utf8')

    model = {}

    for line in f:

        splitLine = line.split(' ')

        word = splitLine[0]

        embedding = np.asarray(splitLine[1:], dtype='float32')

        model[word] = embedding

    print ("Done.",len(model)," words loaded!")

    return model
word_model= loadGloveModel(EMBEDDING_FILE)   

# print (word_model['hello']) # if we want to see an example for a vector
print('Loaded %s word vectors.' % len(word_model))

unknown_words = []

for question in quora_train.question_text_prep:

    for word in question:

        if word not in word_model:

            unknown_words.append(word)

        else: pass
total_term_frequency = Counter(unknown_words)



for word, freq in total_term_frequency.most_common(20):

    print("{}\t{}".format(word, freq))
def get_vector(DataFrame):

    vec_X = []

    i = 0

    for item in DataFrame.question_text_prep_string: 

        

        sentence = pad_punctuation_w_space(item)

        s = np.array([])

        s = []

        if len(sentence)==0:

            s = np.array(word_model['UNK'])

            vec_X.append(s) 

            i += 1

        else:

                for word in sentence.split():

                    if len(s) == 0:

                        try:

                            s = np.array(word_model[word])

                        except: 

                            s = np.array(word_model['UNK'])

                    else:

                        try:

                            s += np.array(word_model[word])

                        except: 

                            s += np.array(word_model['UNK'])         

                vec_X.append(s) 

                i += 1



    return vec_X
vec_X_train=get_vector(quora_train)

vec_X_test=get_vector(quora_test)

quora_train["vector"]=vec_X_train

quora_test["vector"]=vec_X_test
from imblearn.under_sampling import RandomUnderSampler
X = quora_train[['words','characters','vector']] #,'noun_count'

y = quora_train['target']
rus = RandomUnderSampler(return_indices=True, ratio = 0.42)

X_rus, y_rus, id_rus = rus.fit_sample(X, y)



print('indexes:', id_rus)

print(len(id_rus))

print(quora_train.target.value_counts())
quora_undr=quora_train.loc[id_rus]
quora_undr['target'].value_counts(ascending=True).plot(kind='bar')
quora_undr['target'].value_counts(normalize=True)
quora_under_prep = quora_undr
quora_under_prep['noun_count'] = quora_under_prep.question_text.apply(lambda x: noun_count(x))

quora_test['noun_count'] = quora_test.question_text.apply(lambda x: noun_count(x))
quora_under_prep['vector_length']= quora_under_prep['vector'].apply(lambda x: len(x))

quora_test['vector_length']= quora_test['vector'].apply(lambda x: len(x))

quora_test['vector_length'].describe()
quora_best=quora_under_prep
import numpy as np

quora_best["joinvector"]=[np.concatenate((np.array([quora_best["characters"].iloc[i]]),quora_best["vector"].iloc[i]), axis=None) for i in range(len(quora_best))]

quora_best["joinvector_2"]=[np.concatenate((np.array([quora_best["words"].iloc[i]]),quora_best["joinvector"].iloc[i]), axis=None) for i in range(len(quora_best))]

quora_best["joinvector_all"]=[np.concatenate((np.array([quora_best["noun_count"].iloc[i]]),quora_best["joinvector_2"].iloc[i]), axis=None) for i in range(len(quora_best))]
quora_test["joinvector"]=[np.concatenate((np.array([quora_test["characters"].iloc[i]]),quora_test["vector"].iloc[i]), axis=None) for i in range(len(quora_test))]

quora_test["joinvector_2"]=[np.concatenate((np.array([quora_test["words"].iloc[i]]),quora_test["joinvector"].iloc[i]), axis=None) for i in range(len(quora_test))]

quora_test["joinvector_all"]=[np.concatenate((np.array([quora_test["noun_count"].iloc[i]]),quora_test["joinvector_2"].iloc[i]), axis=None) for i in range(len(quora_test))]
X_joinvec=quora_best["joinvector_all"].tolist()
Features = quora_best['joinvector_all']

# Features2=quora_best['vector']
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV





X_train, X_val, y_train, y_val = train_test_split(Features,quora_best['target'],

                                                    train_size=0.7, random_state = 143, stratify=quora_best['target'])
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier



#evaluators:

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from sklearn.model_selection import cross_val_score
X_grid5 = X_train.tolist()

y_grid5 = y_train
from sklearn.ensemble import RandomForestClassifier

Rfr_clf = RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV



param_grid = {'n_estimators':[10,50,100,200],

              'criterion' : ['gini', 'entropy'],

              'class_weight' : [{0:9,1:1},{0:1,1:1},{0:66,1:33}],

              'max_depth' : range(2,10),

              'min_samples_split': range(2,10),

              'min_samples_leaf' : range(2,10),

              'bootstrap': [True,False] }



RFR_clf_gs = RandomizedSearchCV(estimator=Rfr_clf, param_distributions=param_grid, cv=3,

                                verbose=0, n_jobs=-1,scoring='f1')



RFR_clf_gs.fit(X_grid5, y_grid5)

RFR_best=RFR_clf_gs.best_estimator_ 

print("Randomized search process ended")   
y_pred= RFR_best.predict(X_grid5)
# param_grid ={"learning_rate":(0.1,0.5,0.8),

#              'max_depth' : range(2,53,10),

#              'min_samples_split': range(2,53,10),

#              'min_samples_leaf' : range(2,53,10),

#              'n_estimators' : (100,200,300,400) ,

#              'max_features': range(2,303,30),

#              #'random_state': (143),

#              'subsample': (0.1,0.5,0.8,1,2)} 
# GB_clf = GradientBoostingClassifier()

# gs= GridSearchCV(estimator=GB_clf, param_grid=param_grid, cv=3,scoring='f1') # verbose=15, n_jobs=-1

# gs.fit(X_grid5, y_grid5)

# best_model=gs.best_estimator_ 
# y_pred= best_model.predict(X_grid5)
test_Features = quora_test['joinvector_all']

X_test_original=test_Features.tolist()

# y_test_pred= best_model.predict(X_test_original)

y_test_pred= RFR_best.predict(X_test_original)
quora_test_tmp=quora_test
quora_test_tmp["pred"]=y_test_pred #(y_test_pred > delta).astype(int) 

quora_test_tmp1 = quora_test_tmp[['qid','question_text','pred']]

quora_test_tmp1[quora_test_tmp1['pred']==1].sample(10)
sub = pd.read_csv('../input/sample_submission.csv')

out_df = pd.DataFrame({"qid":sub["qid"].values})

out_df['prediction'] = y_test_pred

out_df.to_csv("submission.csv", index=False)
round(out_df['prediction'].value_counts(normalize =True),3)*100