import nltk

import gensim

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import re



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
print('Train length: '+str(len(train['text'])))

print('Test length: '+str(len(test['text'])))

print('Target Distribution: \n'+str(train['author'].value_counts()))
train_text = train['text']

test_text = test['text']

auth_map = {0:'EAP',1:'HPL',2:'MWS'}

auth_map_inv = {'EAP':0,'HPL':1,'MWS':2}

train_tgt = train['author'].map(auth_map_inv)
train_text = [re.sub(r'[^a-z]',' ',text.lower()) for text in train_text]

test_text = [re.sub(r'[^a-z]',' ',text.lower()) for text in test_text]

all_text = train_text + test_text

all_text = [text.split() for text in all_text]
stops = nltk.corpus.stopwords.words('english')

all_text = [ [ word for word in text if word not in stops] for text in all_text ]
porter = nltk.stem.porter.PorterStemmer()

all_text = [ [ porter.stem(word) for word in text] for text in all_text]
min_num = 3



from collections import Counter



counts = Counter()

for text in all_text:

    for word in text:

        counts[word] += 1

to_remove = []

for word in reversed(counts.most_common()):

    if counts[word[0]]<= min_num:

        to_remove.append(word[0])

    else:

        break

print('Total number of words: '+str(len(counts)))

print('Number of words to remove: '+str(len(to_remove)))

#print('Words to remove: \n' + str(to_remove))

all_text = [ [word for word in text if word not in to_remove] for text in all_text]
lens = [len(text) for text in all_text]

plt.hist(lens,bins=100,range=(0,100))

plt.xlabel('Number of Words After Cleaning')

plt.ylabel('Number of Documents')

plt.show()
minlen = 3

n_to_remove = 0

for text in all_text:

    if len(text)<minlen:

        n_to_remove+=1

        

print('Number of texts smaller than minimum: '+str(n_to_remove) + ' of '+str(len(all_text)))
all_text = [text for text in all_text if len(text)>=minlen]
dictionary = gensim.corpora.Dictionary(all_text)

word_vec = [ dictionary.doc2bow(text) for text in all_text ]
tfidf = gensim.models.TfidfModel(word_vec)

tfidf_vec = [tfidf[vec] for vec in word_vec]

lsi = gensim.models.lsimodel.LsiModel(corpus=tfidf_vec,

                                      id2word=dictionary, 

                                      num_topics=50)



lda = gensim.models.ldamodel.LdaModel(corpus=tfidf_vec,

                                      id2word=dictionary,

                                      num_topics=50,passes=10)

#                                      update_every=1,

#                                      chunksize=5000,

#                                      passes=10)
print('LSA Results')

lsi.print_topics(50)

print('LDA Results')

lda.print_topics(50)
# mod gives the mapping from tfidf to the 50-feature space

def prepare_text(data,mod):

    data = [re.sub(r'[^a-z]',' ',text.lower()) for text in data]

    data = [text.split() for text in data]

    data = [ [ word for word in text if word not in stops] for text in data ]

    data = [ [ porter.stem(word) for word in text] for text in data] 

    data = [dictionary.doc2bow(text) for text in data]

    data = [tfidf[text] for text in data ]

    data = [mod[text] for text in data]



    data_arr = []

    for text in data:

        data_arr.append([0.]*50)

        for word in text:

            data_arr[-1][word[0]] = word[1]



    return np.array(data_arr)
train_prep = prepare_text(train_text,lsi)

test_prep = prepare_text(test_text,lsi)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

import sklearn.metrics as metrics





# Get class weights: The data set is imbalanced



weights = np.max(train_tgt.value_counts())/train_tgt.value_counts()

print(weights)



c_weight = {}

for i in range(len(weights)):

    c_weight[weights.index[i]] = weights[i]

    

# This is just the same as scoring='neg_log_loss'

logloss_score = metrics.make_scorer(metrics.log_loss,needs_proba=True,greater_is_better=False)



# Random forest with some random params

rfc = RandomForestClassifier(max_depth=10,min_samples_leaf=10,n_estimators=50,

                             class_weight=c_weight)



# Cross validate

scores = cross_validate(rfc,train_prep,np.array(train_tgt),cv=5,

                        scoring=logloss_score)

scores
from sklearn.model_selection import GridSearchCV



pars = {'max_depth':[5,10,15],'min_samples_leaf':[5,10,15],'n_estimators':[50]}

gcv = GridSearchCV(rfc,param_grid=pars,scoring=logloss_score,cv=5)

gcv.fit(train_prep,np.array(train_tgt))

print(gcv.best_params_)

print(gcv.cv_results_['mean_test_score'])

print(gcv.cv_results_['mean_train_score'])
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

all_text_joined = [ ' '.join(text) for text in all_text]



cvec = CountVectorizer(ngram_range=(1,3),min_df=2)

cvec.fit(all_text_joined)




def prepare_text_counts(data):

    data = [re.sub(r'[^a-z]',' ',text.lower()) for text in data]

    data = [text.split() for text in data]

    data = [ [ word for word in text if word not in stops] for text in data ]

    data = [ [ porter.stem(word) for word in text] for text in data] 

    data = [ ' '.join(text) for text in data]

    data = cvec.transform(data)

    return data

    

    
train_text = train['text']

test_text = test['text']

train_prep = prepare_text_counts(train_text)

test_prep = prepare_text_counts(test_text)
from sklearn.model_selection import validation_curve



mnb = MultinomialNB(alpha=1)

vals = np.logspace(-1,2,50)

train_scores,valid_scores = validation_curve(mnb,train_prep,

                                             np.array(train_tgt),'alpha',

                                             vals,cv=10,scoring=logloss_score)



train_mean = -np.mean(train_scores,axis=1)

valid_mean = -np.mean(valid_scores,axis=1)

train_std = np.std(train_scores,axis=1)

valid_std = np.std(valid_scores,axis=1)



fig = plt.figure(1,(8,8))

plt.plot(vals,train_mean,color='b',label='Training Scores')

plt.fill_between(vals,train_mean-train_std,train_mean+train_std,facecolor='b',alpha=0.3)



plt.plot(vals,valid_mean,color='r',label='Validation Scores')

plt.fill_between(vals,valid_mean-valid_std,valid_mean+valid_std,facecolor='r',alpha=0.3)



plt.legend()

plt.ylabel('Log-Loss Score')

plt.xlabel('Alpha')

plt.xscale('log')

plt.title('Validation Curves for Multinomial Naive Bayes')

plt.show()