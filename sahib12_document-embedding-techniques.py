# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import gensim
from gensim import utils
import sys
import nltk
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk import download
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%matplotlib inline
#model Google News, run once to download pre-trained vectors
!wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
download('punkt')
download('stopwords')# stopwords

stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc
# Fetch ng20 dataset
ng20 = fetch_20newsgroups(subset='all',
                          remove=('headers', 'footers', 'quotes'))
# text and ground truth labels
texts, y = ng20.data, ng20.target


print(type(texts))
corpus=[preprocess(text) for text in texts]
def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)
corpus,texts,labels=filter_docs(corpus,texts,y,lambda doc: (len(doc) != 0))

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
   
    return np.mean(word2vec_model[doc], axis=0)
def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)
corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: has_vector_representation(model, doc))
x =[]
for doc in corpus: #look up each doc in model
    x.append(document_vector(model, doc))
X = np.array(x) #list to array
np.save('documents_vectors.npy', X)  #np.savetxt('documents_vectors.txt', X)
np.save('labels.npy', y)             #np.savetxt('labels.txt', y)
X.shape, len(y)
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
test.head()
submission.head()
text_train,y=list(train.text),list(train.target)
def preprocess_twitter(text):
    text = text.lower()
    doc = word_tokenize(text)
    return doc

corpus_train=[preprocess_twitter(text) for text in text_train]
def document_vector_twitter(word2vec_model, doc):
    # remove out-of-vocabulary words

    docs=[]
    for word in doc:
        if word in word2vec_model.vocab:
            docs.append(word)

        else:
            docs.append('fire')
   # return np.mean(word2vec_model[docs], axis=0)
# because some words in a sentence are more important than others so by averaging we will decrease their importance 
# so just sum it to preserve the importance
    return np.sum(word2vec_model[docs], axis=0)
x_train =[]
for doc in corpus_train: #look up each doc in model
    x_train.append(document_vector_twitter(model, doc))
print(type(x_train))
print(type(x_train[0]))
print(len(x_train))
print((x_train[0].shape))
from sklearn.linear_model import LogisticRegression
model_final = LogisticRegression(C=4,max_iter=3000)
model_final.fit(x_train,y)
test_text=list(test.text)
print(len(test_text))
# preprocessing
# Not done complete preprocessing like removing stopwords and removing numbers etc
# because removing these words actually removes those docs that just have one word and that too a stopword so once if we remove that word our doc will be [] and our model cant make doc2vec for this

corpus_test=[preprocess_twitter(text) for text in test_text]
len(corpus_test)
x_test =[]
for doc in corpus_test: #look up each doc in model
    x_test.append(document_vector_twitter(model, doc))
print(len(x_test))
prediction=list(model_final.predict(x_test))
submission['target']=prediction

submission.to_csv('submission1.csv', index=False)
submission.head(10)
import re
import string
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# function for removing stopwords

def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return words
from sklearn.feature_extraction.text import TfidfVectorizer
corpus_tester = [
    'This is the first document deed.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


#preprocessing
text_train=[clean_text(text) for text in corpus_tester]


# Tokenizing 
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

text_train=[tokenizer.tokenize(text) for text in text_train]


# stopwords removal
text_train=[remove_stopwords(text) for text in text_train]


#lemmatization

lmtzr= WordNetLemmatizer() 

# After preprocessing, the text format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

text_train=[combine_text(text) for text in text_train]

# print(text_train)




vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix= vectorizer.fit_transform(text_train)




def tester(word2vec,doc):
    trainer=[] # List that will contain document embedding for all docs
    for i in range(len(ls)):
            final_doc=[] # list that will contain document embedding for a single document

            # getting tfidf matrix for given document
            first_vector_tfidfvectorizer=tfidf_matrix[i]

            # place tf-idf values in a pandas data frame
            df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
            df.sort_values(by=["tfidf"],ascending=False)

#             print(df)

            for j in range(len(ls[i])):

                word=(ls[i][j]) # storing current word of present document
                
                word=lmtzr.lemmatize(word) # lemmatizing current word

                word_vector=model[word] # getting array of shape (300,) from word2vec model of given word

                tf_idf_vector=df['tfidf'][str(word)] # getting TfIdf score that current word from TfIdf dataframe created above
                
               
                word_vector=word_vector*tf_idf_vector #mutiplying TfIdf score of a present word with complete array we got from word2vec model
                

                final_doc.append(word_vector)
                
              

            final_doc=np.asarray(final_doc) # converting list of word2vec to doc2vec
            final_doc=np.sum(final_doc,axis=0) 
            
            trainer.append(final_doc)
    return trainer
    


#before passing it to function            
ls=[]
for i in range(len(text_train)):
    ls.append(text_train[i].split(' '))



f=tester(model,ls)
y=list(train.target)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus_tester = list(train.text)


#preprocessing
text_train=[clean_text(text) for text in corpus_tester]


# Tokenizing 
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

text_train=[tokenizer.tokenize(text) for text in text_train]


# stopwords removal
text_train=[remove_stopwords(text) for text in text_train]


#lemmatization
lmtzr= WordNetLemmatizer() 


# After preprocessing, the text format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

text_train=[combine_text(text) for text in text_train]






vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix= vectorizer.fit_transform(text_train)




def tester(word2vec,doc,target_var):
    trainer=[]
    for i,y in zip(range(len(ls)),target_var):
            final_doc=[]

            # get the first vector out (for the first document)
            first_vector_tfidfvectorizer=tfidf_matrix[i]

            # place tf-idf values in a pandas data frame
            df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
            df.sort_values(by=["tfidf"],ascending=False)
            
            

            for j in range(len(ls[i])):

                word=(ls[i][j])

                word=lmtzr.lemmatize(word)
                
                try:

                    word_vector=model[word]
                    tf_idf_vector=df['tfidf'][str(word)]

                    word_vector=word_vector*tf_idf_vector

                    final_doc.append(word_vector)
                except :
                    word=str('fire') # I am considering that those sentence which were skipped and whose documnt shape was ()
                    # I have considered that those sentences as True comments and all those sentences consist of single word 'Fire'
                    # Because It was th most used as we have seen from wordcloud

                    word=lmtzr.lemmatize(word)
                        
                    word_vector=model[word]
                        
                    tf_idf_vector=df['tfidf'][str(word)]

                    word_vector=word_vector*tf_idf_vector

                    final_doc.append(word_vector)
                    
                    
                    

                        

                        
            final_doc=np.asarray(final_doc) # converting list of word2vec to doc2vec
            final_doc=np.sum(final_doc,axis=0) 
            trainer.append(final_doc)


    return trainer
    


#before passing it to function            
ls=[]
for i in range(len(text_train)):
    ls.append(text_train[i].split(' '))


f=tester(model,ls,y)
for i,j in  enumerate(f):
    if j.shape!=(300,):
        print(i," and it's shape is ",j.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus_tester = list(train.text)


#preprocessing
text_train=[clean_text(text) for text in corpus_tester]


# Tokenizing 
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

text_train=[tokenizer.tokenize(text) for text in text_train]


# stopwords removal
text_train=[remove_stopwords(text) for text in text_train]


#lemmatization
lmtzr= WordNetLemmatizer() 


# After preprocessing, the text format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

text_train=[combine_text(text) for text in text_train]






vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix= vectorizer.fit_transform(text_train)




def tester(word2vec,doc,target_var):
    trainer=[]
    for i,y in zip(range(len(ls)),target_var):
            final_doc=[]

            # get the first vector out (for the first document)
            first_vector_tfidfvectorizer=tfidf_matrix[i]

            # place tf-idf values in a pandas data frame
            df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
            df.sort_values(by=["tfidf"],ascending=False)
            
            

            for j in range(len(ls[i])):

                word=(ls[i][j])

                word=lmtzr.lemmatize(word)
                
                try:

                    word_vector=model[word]
                    tf_idf_vector=df['tfidf'][str(word)]

                    word_vector=word_vector*tf_idf_vector

                    final_doc.append(word_vector)
                except :
                    word=str('fire') # I am considering that those sentence which were skipped and whose documnt shape was ()
                    # I have considered that those sentences as True comments and all those sentences consist of single word 'Fire'
                    # Because It was th most used as we have seen from wordcloud

                    word=lmtzr.lemmatize(word)
                        
                    word_vector=model[word]
                        
                    tf_idf_vector=df['tfidf'][str(word)]

                    word_vector=word_vector*tf_idf_vector

                    final_doc.append(word_vector)
                    
                    
                    

                        

                        
            final_doc=np.asarray(final_doc) # converting list of word2vec to doc2vec
            final_doc=np.sum(final_doc,axis=0) 
            trainer.append(final_doc)


    return trainer
    


#before passing it to function            
ls=[]
for i in range(len(text_train)):
    ls.append(text_train[i].split(' '))


f=tester(model,ls,y)
for i,j in  enumerate(f):
    if j.shape!=(300,):
        print(i," and it's shape is ",j.shape)

from sklearn.linear_model import LogisticRegression
model_final1 = LogisticRegression(C=4,max_iter=5000)
model_final1.fit(f,y)
test_text=list(test.text)
print(len(test_text))


test_text=list(test.text)


#preprocessing
test_text=[clean_text(text) for text in test_text]


# Tokenizing 
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

test_text=[tokenizer.tokenize(text) for text in test_text]


# stopwords removal
test_text=[remove_stopwords(text) for text in test_text]


#lemmatization
lmtzr= WordNetLemmatizer() 


# After preprocessing, the text format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

test_text=[combine_text(text) for text in test_text]






vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix_test= vectorizer.fit_transform(test_text)




def tester_test(word2vec,doc):
    trainer=[]
    for i in range(len(ls)):
            final_doc=[]

            # get the first vector out (for the first document)
            first_vector_tfidfvectorizer=tfidf_matrix_test[i]

            # place tf-idf values in a pandas data frame
            df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
            df.sort_values(by=["tfidf"],ascending=False)
            
            

            for j in range(len(ls[i])):

                word=(ls[i][j])

                word=lmtzr.lemmatize(word)
                
                try:

                    word_vector=model[word]
                    tf_idf_vector=df['tfidf'][str(word)]

                    word_vector=word_vector*tf_idf_vector

                    final_doc.append(word_vector)
                except :
                    word=str('fire') # I am considering that those sentence which were skipped and whose documnt shape was ()
                    # I have considered that those sentences as True comments and all those sentences consist of single word 'Fire'
                    # Because It was th most used as we have seen from wordcloud

                    word=lmtzr.lemmatize(word)
                        
                    word_vector=model[word]
                        
                    tf_idf_vector=df['tfidf'][str(word)]

                    word_vector=word_vector*tf_idf_vector

                    final_doc.append(word_vector)
                    
                    
                    

                        

                        
            final_doc=np.asarray(final_doc) # converting list of word2vec to doc2vec
            final_doc=np.sum(final_doc,axis=0) 
            trainer.append(final_doc)


    return trainer
    


#before passing it to function            
ls=[]
for i in range(len(test_text)):
    ls.append(test_text[i].split(' '))


f_test=tester_test(model,ls)
prediction=list(model_final1.predict(f_test))
submission['target']=prediction

submission.to_csv('submission.csv', index=False)
submission.head(10)

