#Importing libraries
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
%matplotlib inline
#importing dataset
messages = pd.read_csv("../input/spamraw.csv")
messages.head()
messages.describe()
#Let's add a column of length to our data
messages['length'] = messages['text'].apply(len)
messages.head()
#let's visualize
messages['length'].plot.hist(bins=50)
messages['length'].describe()
#Creating histograms for Ham,Spam
messages.hist(column = 'length',by = 'type',bins = 60,figsize = (12,4))
#Creating Function
from nltk.corpus import stopwords
def text_process(text):
    """
    1.Remove punc
    2.Remove stop words
    3.Return list of clean text words
    """
    nonpunc = [char for char in text if char not in string.punctuation]
    nonpunc = ''.join(nonpunc)
    return[word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
messages['text'].head(5).apply(text_process)
#Step 1
from sklearn.feature_extraction.text import  CountVectorizer
bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['text'])
print(len(bow_transformer.vocabulary_)) #This is going to print total no. of vocab words
#lets check transform vocabulary for one message
#bow is Bag of Words
mess4 = messages['text'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
messages_bow = bow_transformer.transform(messages['text'])
print('shape of sparse matrix :',messages_bow.shape)
#Check amount of non-zero occurences
messages_bow.nnz
#step2 & step3
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
#lets check 
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

#lets build our model
from sklearn.model_selection import train_test_split
msg_train,msg_test,type_train,type_test = train_test_split(messages['text'],
                                                             messages['type'],
                                                             test_size = 0.3)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer = text_process)),
     ('tfidf',TfidfTransformer()),
      ('classifier',MultinomialNB())])
#Now we can directly pass our message text and pipeline will do all our pre-processing for us
pipeline.fit(msg_train,type_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(type_test,predictions))
