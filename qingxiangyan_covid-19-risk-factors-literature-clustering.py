# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # regular expressions

import nltk

import random # for random sampling 300 abstracts for annotation. Can be removed later

from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, parse_dates=['publish_time'], dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
meta_df.info()
meta_df_new = meta_df[meta_df['publish_time'] >= '2019-01-01']

meta_df_new.info()
abstracts_new = meta_df_new[['cord_uid', 'abstract']].dropna() # create new df with only id and abstract

about_coronavirus = abstracts_new['abstract'].apply(lambda x: ('coronavirus' in x.lower() or 'covid' in x.lower())) # create condition that abstract contains 'coronavirus' or 'COVID'

abstracts_new = abstracts_new[about_coronavirus] # filter abstracts based on about_coronavirus condition

abstracts_new.info()
abstracts_new.head()
abstracts_new['tokens'] = abstracts_new['abstract'].apply(lambda x: re.sub('[^a-zA-z\s]',' ',x)) # remove punctuation and numbers from abstract text

abstracts_new.head()
abstracts_new['tokens'] = abstracts_new['tokens'].apply(lambda x: word_tokenize(x.lower())) #tokenize lowercase words

abstracts_new.head()
# define a function to remove stop words from a list of words

def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words
abstracts_new['tokens'] = abstracts_new['tokens'].apply(lambda x: remove_stopwords(x)) #remove stopwords

abstracts_new.head()
# define a function to remove the word 'abstract' from a list of words

def remove_abstract(text):

    words = [w for w in text if w != 'abstract']

    return words
abstracts_new['tokens'] = abstracts_new['tokens'].apply(lambda x: remove_abstract(x)) #remove the word 'abstract' from token list

abstracts_new.head()
# define a function to lematize a list of tokens

lemmatizer = WordNetLemmatizer() # instantiate a lemmatizer

def lemmatize_tokens(tokens):

    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return lemmatized_tokens
abstracts_new['tokens'] = abstracts_new['tokens'].apply(lambda x: lemmatize_tokens(x)) #lemmatize words

abstracts_new.head()
abstracts_new['bow'] = abstracts_new['tokens'].apply(lambda x: Counter(x)) # create new column called 'bow' that translates lemmatized tokens into bag of words 

abstracts_new.head()
# define a function that converts a list of tokens into n-grams

def getNGrams(tokens, n):

    return [tokens[i : i + n] for i in range(len(tokens) - (n - 1))]
abstracts_new['2-grams'] = abstracts_new['tokens'].apply(lambda x: getNGrams(x, 2))

abstracts_new.head()
abstracts_new.info()
# read in data

model_data_df = pd.read_csv('../input/covid19-data-for-modeling/Metadata_for_modeling.csv',encoding = "ISO-8859-1")

# get training and test samples

train_test = model_data_df[model_data_df['split_label'].notna()]

train_test.info()
# Training set

train = train_test[train_test['split_label']=="Training"]

train.info()
# Test set

test = train_test[train_test['split_label']=="Test"]

test.info()
train_test_tmp = train_test



#train Vectorizer the entire training+test set

words = set(nltk.corpus.words.words())



vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 2000) 

                             

vectorizer.fit(train_test_tmp['tokens'])



#Vectorizing training set

x_train = vectorizer.transform(train['tokens'])

x_train = x_train.toarray()



# Top words in the trianing set

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_train.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')



#vectorizing test set

x_test = vectorizer.transform(test['tokens'])

x_test = x_test.toarray()



# Top words in the test set

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_test.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
# Train a simple logistic regression

logisticCV = LogisticRegressionCV(cv=5, random_state=19, max_iter = 10000).fit(x_train, train['risk_factor'])

# Training performance

bow_lr_train = logisticCV.score(x_train, train['risk_factor']) 

print(bow_lr_train)
# Test performance

bow_lr_test = logisticCV.score(x_test, test['risk_factor'])  

print(bow_lr_test)
# Train a random forest model



# Select hyperparameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  

# param_grid = {

#                 'n_estimators': [50,100,150,200,500],

#                 'max_depth': list(range(1,20, 2))

#             }



# clf = RandomForestClassifier(random_state=19)



# grid_clf = GridSearchCV(clf, param_grid, cv=5)



# grid_clf.fit(x_train, train['risk_factor'])



# grid_clf.best_estimator_



forestCV = RandomForestClassifier(n_estimators = 100, max_depth=17, random_state=19) 

forestCV = forestCV.fit(x_train, train['risk_factor'])



# Training performance

bow_rf_train  = forestCV.score(x_train, train['risk_factor']) 

print(bow_rf_train)



# Test performance

bow_rf_test = forestCV.score(x_test, test['risk_factor']) 

print(bow_rf_test)
# Train an MLP model





# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  



# param_grid = {'hidden_layer_sizes': [(20,10),(10,5),(4,2)]}



# clf = MLPClassifier(random_state=19, max_iter = 10000)



# grid_clf = GridSearchCV(clf, param_grid, cv=5)



# grid_clf.fit(x_train, train['risk_factor'])



# grid_clf.best_estimator_





NN = MLPClassifier(solver='lbfgs', alpha=0.0001,activation='relu',

                    hidden_layer_sizes=(4, 2), random_state=19, max_iter = 1000)



NN = NN.fit(x_train, train['risk_factor'])



# Training performance

bow_mlp_train = NN.score(x_train, train['risk_factor'])

print(bow_mlp_train)
# Test performance

bow_mlp_test = NN.score(x_test, test['risk_factor'])  

print(bow_mlp_test)
train_test_tmp = train_test



#train Vectorizer the entire training+test set

words = set(nltk.corpus.words.words())



vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             ngram_range = (1,2),    # <- include 1 and 2-grams

                             max_features = 2000) 

                             

vectorizer.fit(train_test_tmp['tokens'])



#Vectorizing training set

x_train = vectorizer.transform(train['tokens'])

x_train = x_train.toarray()



# Top words in the trianing set

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_train.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
#vectorizing test set

x_test = vectorizer.transform(test['tokens'])

x_test = x_test.toarray()



# Top words in the test set

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_test.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
# Train a logistic regression

logisticCV = LogisticRegressionCV(cv=5, random_state=19, max_iter = 10000).fit(x_train, train['risk_factor'])

# Training performance

ngram_lr_train = logisticCV.score(x_train, train['risk_factor']) 

print(ngram_lr_train)
# Test performance

ngram_lr_test = logisticCV.score(x_test, test['risk_factor'])  

print(ngram_lr_test)
# Train a random forest model



# Select hyperparameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  

# param_grid = {

#                 'n_estimators': [50,100,150,200,500],

#                 'max_depth': list(range(1,20, 2))

#             }



# clf = RandomForestClassifier(random_state=19)



# grid_clf = GridSearchCV(clf, param_grid, cv=5)



# grid_clf.fit(x_train, train['risk_factor'])



# grid_clf.best_estimator_



forestCV = RandomForestClassifier(n_estimators = 100,max_depth=13, random_state=19) 

forestCV = forestCV.fit(x_train, train['risk_factor'])



# Training performance

ngram_rf_train  = forestCV.score(x_train, train['risk_factor']) 

print(ngram_rf_train)
# Test performance

ngram_rf_test = forestCV.score(x_test, test['risk_factor']) 

print(ngram_rf_test)
# Train an MLP model



# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  



#param_grid = {'hidden_layer_sizes': [(20,10),(10,5),(4,2)]}



#clf = MLPClassifier(random_state=19, max_iter = 10000)



#grid_clf = GridSearchCV(clf, param_grid, cv=5)



#grid_clf.fit(x_train, train['risk_factor'])



#grid_clf.best_estimator_



NN = MLPClassifier(solver='lbfgs', alpha=0.0001,activation='relu',

                    hidden_layer_sizes=(10, 5), random_state=19, max_iter = 1000)



NN = NN.fit(x_train, train['risk_factor'])



# Training performance

ngram_mlp_train = NN.score(x_train, train['risk_factor'])

print(ngram_mlp_train)
# Test performance

ngram_mlp_test = NN.score(x_test, test['risk_factor'])  

print(ngram_mlp_test)
train_test_tmp = train_test



#train Vectorizer the entire training+test set

words = set(nltk.corpus.words.words())



vectorizer = TfidfVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             ngram_range = (1,1),

                             max_features = 2000)



vectorizer.fit(train_test_tmp['tokens'])



#Vectorizing training set

x_train = vectorizer.transform(train['tokens'])

x_train = x_train.toarray()



word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_train.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
#vectorizing test set

x_test = vectorizer.transform(test['tokens'])

x_test = x_test.toarray()



# Top words in the test set

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_test.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
# Train a simple logistic regression

logisticCV = LogisticRegressionCV(cv=5, random_state=19, max_iter = 10000).fit(x_train, train['risk_factor'])

# Training performance

Tfidf_bow_lr_train = logisticCV.score(x_train, train['risk_factor']) 

print(Tfidf_bow_lr_train)
# Test performance

Tfidf_bow_lr_test = logisticCV.score(x_test, test['risk_factor'])  

print(Tfidf_bow_lr_test)
# Train a random forest model



# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  

# param_grid = {

#                 'n_estimators': [50,100,150,200,500],

#                 'max_depth': list(range(1,20, 2))

#             }



# clf = RandomForestClassifier(random_state=19)



# grid_clf = GridSearchCV(clf, param_grid, cv=5)



# grid_clf.fit(x_train, train['risk_factor'])



# grid_clf.best_estimator_

forestCV = RandomForestClassifier(n_estimators = 50,max_depth=9, random_state=19) 

forestCV = forestCV.fit(x_train, train['risk_factor'])



# Training performance

Tfidf_bow_rf_train  = forestCV.score(x_train, train['risk_factor']) 

print(Tfidf_bow_rf_train)
# Test performance

Tfidf_bow_rf_test = forestCV.score(x_test, test['risk_factor']) 

print(Tfidf_bow_rf_test)
# Train an MLP model

# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  



#param_grid = {'hidden_layer_sizes': [(20,10),(10,5),(4,2)]}



#clf = MLPClassifier(random_state=19, max_iter = 10000)



#grid_clf = GridSearchCV(clf, param_grid, cv=5)



#grid_clf.fit(x_train, train['risk_factor'])



#grid_clf.best_estimator_



NN = MLPClassifier(solver='lbfgs', alpha=0.0001,activation='relu',

                    hidden_layer_sizes=(10, 5), random_state=19, max_iter = 1000)



NN = NN.fit(x_train, train['risk_factor'])



# Training performance

Tfidf_bow_mlp_train = NN.score(x_train, train['risk_factor'])

print(Tfidf_bow_mlp_train)
# Test performance

Tfidf_bow_mlp_test = NN.score(x_test, test['risk_factor'])  

print(Tfidf_bow_mlp_test)
train_test_tmp = train_test



#train Vectorizer the entire training+test set

words = set(nltk.corpus.words.words())



vectorizer = TfidfVectorizer(analyzer = "word",   

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = None,   

                             ngram_range = (1,2),  

                             max_features = 2000)



vectorizer.fit(train_test_tmp['tokens'])



#Vectorizing training set

x_train = vectorizer.transform(train['tokens'])

x_train = x_train.toarray()



word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_train.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
#vectorizing test set

x_test = vectorizer.transform(test['tokens'])

x_test = x_test.toarray()



# Top words in the test set

word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_test.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
# Train a simple logistic regression

logisticCV = LogisticRegressionCV(cv=5, random_state=19, max_iter = 10000).fit(x_train, train['risk_factor'])

# Training performance

Tfidf_ngram_lr_train = logisticCV.score(x_train, train['risk_factor']) 

print(Tfidf_ngram_lr_train)
# Test performance

Tfidf_ngram_lr_test = logisticCV.score(x_test, test['risk_factor'])  

print(Tfidf_ngram_lr_test)
# Train a random forest model



# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  

# param_grid = {

#                 'n_estimators': [50,100,150,200,500],

#                 'max_depth': list(range(1,20, 2))

#             }



# clf = RandomForestClassifier(random_state=19)



# grid_clf = GridSearchCV(clf, param_grid, cv=5)



# grid_clf.fit(x_train, train['risk_factor'])



# grid_clf.best_estimator_



forestCV = RandomForestClassifier(n_estimators = 50,max_depth=11, random_state=19) 

forestCV = forestCV.fit(x_train, train['risk_factor'])



# Training performance

Tfidf_ngram_rf_train  = forestCV.score(x_train, train['risk_factor']) 

print(Tfidf_ngram_rf_train)

# Test performance

Tfidf_ngram_rf_test = forestCV.score(x_test, test['risk_factor']) 

print(Tfidf_ngram_rf_test)
# Train an MLP model

# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  



#param_grid = {'hidden_layer_sizes': [(20,10),(10,5),(4,2)]}



#clf = MLPClassifier(random_state=19, max_iter = 10000)



#grid_clf = GridSearchCV(clf, param_grid, cv=5)



#grid_clf.fit(x_train, train['risk_factor'])



#grid_clf.best_estimator_



NN = MLPClassifier(solver='lbfgs', alpha=0.0001,activation='relu',

                    hidden_layer_sizes=(10, 5), random_state=19, max_iter = 1000)



NN = NN.fit(x_train, train['risk_factor'])



# Training performance

Tfidf_ngram_mlp_train = NN.score(x_train, train['risk_factor'])

print(Tfidf_ngram_mlp_train)
# Test performance

Tfidf_ngram_mlp_test = NN.score(x_test, test['risk_factor'])  

print(Tfidf_ngram_mlp_test)
# summary table for training performance:

train_res = [['Bow',bow_lr_train, bow_rf_train, bow_mlp_train],

             ['1,2-gram', ngram_lr_train, ngram_rf_train, ngram_mlp_train],

             ['Tfidf-Bow', Tfidf_bow_lr_train, Tfidf_bow_rf_train, Tfidf_bow_mlp_train], 

             ['Tfidf-1,2-gram', Tfidf_ngram_lr_train, Tfidf_ngram_rf_train, Tfidf_ngram_mlp_train]]

train_res = pd.DataFrame(train_res, columns = ['Features','Logistic','Random Forest','MLP'])
train_res
# summary table for testing performance:

test_res = [['Bow',bow_lr_test, bow_rf_test, bow_mlp_test],

             ['1,2-gram', ngram_lr_test, ngram_rf_test, ngram_mlp_test],

             ['Tfidf-Bow', Tfidf_bow_lr_test, Tfidf_bow_rf_test, Tfidf_bow_mlp_test], 

             ['Tfidf-1,2-gram', Tfidf_ngram_lr_test, Tfidf_ngram_rf_test, Tfidf_ngram_mlp_test]]

test_res = pd.DataFrame(test_res, columns = ['Features','Logistic','Random Forest','MLP'])
test_res
# Merge previously annotated labels back to the collection of filtered abstracts: 



labels = train_test[["cord_uid","risk_factor"]]



abstracts_new_merged = pd.merge(abstracts_new,labels, how = 'left', on = ['cord_uid'])

abstracts_new_merged.info()

abstracts_new_merged.head()
# Initialize vectorizer on the entire dataset



train_test_tmp = abstracts_new_merged



# Note： The direct use of abstracts_new_merged['token'] dosen't work so I repeated the preoprocessing here. 



train_test_tmp['newtokens'] = train_test_tmp['abstract'].apply(lambda x: " ".join(x.lower() for x in x.split()))



# Removing punctuation

train_test_tmp['newtokens'] = train_test_tmp['newtokens'].apply(lambda x: re.sub(r'[^a-zA-z\w\s]','',x))



# Stop word removal

stop = stopwords.words('english')

stop.append('abstract')

train_test_tmp['newtokens'] = train_test_tmp['newtokens'].apply(lambda x: " ".join(w for w in x.split() if not w in stop))



#Stemming

lemmatizer = WordNetLemmatizer()

train_test_tmp['newtokens'] = train_test_tmp['newtokens'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))





#train Vectorizer the entire training+test set

words = set(nltk.corpus.words.words())



vectorizer = TfidfVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             ngram_range = (1,2),    # <- indicate 1 and 2-grams

                             max_features = 2000)



vectorizer.fit(train_test_tmp['newtokens'])



# This is our new training set

train = train_test_tmp[train_test_tmp['risk_factor'].notna()]

train.info()





# This is our new test set that includes all the unlabeled abstracts

test = train_test_tmp[train_test_tmp['risk_factor'].isna()]

test.info()




#Vectorizing training set

x_train = vectorizer.transform(train['newtokens'])

x_train = x_train.toarray()



word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_train.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')





#vectorizing testing set

x_test = vectorizer.transform(test['newtokens'])

x_test = x_test.toarray()



word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(x_test.sum(axis=0))})

word_count.sort_values('count', ascending=False).set_index('word')[:30].sort_values('count', ascending=True).plot(kind='barh')
# Train the final MLP model

# Select hyper parameter through cross-validation. To shorten the process time of this notebook, this step is only run once and the results will be used directly.  



# param_grid = {'hidden_layer_sizes': [(50,20),(10,5),(10,2)]}

# clf = MLPClassifier(random_state=19, max_iter = 10000)



# grid_clf = GridSearchCV(clf, param_grid, cv=5)



# grid_clf.fit(x_train, train['risk_factor'])



# grid_clf.best_estimator_



NN = MLPClassifier(solver='adam', alpha=0.0001,activation='relu',

                    hidden_layer_sizes=(10, 5), random_state=19, max_iter = 1000)



final_model = NN.fit(x_train, train['risk_factor'])



# Training performance

Tfidf_ngram_mlp_final = final_model.score(x_train, train['risk_factor'])

print(Tfidf_ngram_mlp_final)
# Merge results and obtain all risk-factor related articles

results = final_model.predict(x_test)



output = pd.DataFrame( data={"cord_uid":test["cord_uid"], "Predicted_label":results})



output.info()
tmp = pd.merge(abstracts_new_merged,output,how = 'left', on = ['cord_uid'])



RiskFactor_df = tmp[(tmp['Predicted_label'] == "Yes" ) | (tmp['risk_factor']== "Yes")]



RiskFactor_df.info()
RiskFactor_df.head()
LDA_data = RiskFactor_df['tokens']



# Reformatting tokens for LDA



def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



texts = list(sent_to_words(LDA_data))



dictionary = corpora.Dictionary(texts)

dict(list(dictionary.token2id.items())[0:10])
corpus = [dictionary.doc2bow(item) for item in texts]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                            id2word=dictionary,

                                            num_topics=10,

                                            random_state=100,

                                            update_every=1,

                                            chunksize=100,

                                            passes=10,

                                            alpha='auto',

                                            per_word_topics=True)

# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
import pyLDAvis

import pyLDAvis.gensim  

# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, R = 50)

vis