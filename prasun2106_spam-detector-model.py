import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import nltk

from nltk.corpus import stopwords

import string
messages  = pd.read_csv('../input/SMSSpamCollection', sep = '\t', names = ['label', 'message'] )
messages.head()
messages.info()
messages.describe()
messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)
messages.head()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
messages['label1'] = label_encoder.fit_transform(messages['label'])
messages.head()
messages = messages.drop('label', axis = 1)
messages = messages.rename(columns = {'label1' : 'label'})
messages.head()
sns.distplot(messages['length'], bins = 100)
messages.length.describe()
messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column = 'length', by = 'label', bins = 50,figsize = (12,4 ))
#nltk.download_shell()
a = string.punctuation
b = pd.DataFrame(stopwords.words('english'))
b.count()
b
sample  = 'Sample!!@@#$ message&&&&'
nopunc = [c for c in sample if c not in a]

nopunc
sample2 = ''.join(nopunc)

sample2
messages['stripped_punc']= messages['message'].apply(lambda x : ''.join ([c for c in x if c not in a]))
messages.head()
sample_message = 'this is a sample message'
def remove_Stopwords (list):

    list = [word for word in list.split() if word.lower() not in stopwords.words('english')]

    

    return list
remove_Stopwords(sample_message)
def text_processing(mess):

    """

    1. Remove Punctuation

    2. Convert to Lowercase

    3. Remove Stopwords

    4. Return list of clean texts

    

    """

    nopunc = [c for c in mess if c not in string.punctuation]

    nopunc = ''.join(nopunc)

    clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_mess

    #removing stopwords 2    
messages['message'].head(5).apply(text_processing) 
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer= text_processing).fit(messages['message'])
type(bow_transformer)
print(len(bow_transformer.vocabulary_))
mess4 = messages['message'][3]
mess4
bow_transformer.transform([mess4])
bow4 = bow_transformer.transform([mess4])
print (bow4.shape)
print(bow4)
bow_transformer.get_feature_names()[9554]
bow_messages = bow_transformer.transform(messages['message'])
bow_messages.nnz
sparsity = bow_messages.nnz*100/(bow_messages.shape[0]* bow_messages.shape[1])
sparsity
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(bow_messages)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
tfidf = tfidf_transformer.transform(bow_messages)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model.fit(tfidf, messages['label'])
spam_detect_model.predict(tfidf4)[0]
all_pred = spam_detect_model.predict(tfidf)
all_pred
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.3)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer = text_processing)),

    ('tfidf', TfidfTransformer()),

    ('classifier', RandomForestClassifier())

    

    

])
pipeline.fit(msg_train, label_train)
all_pred_rf = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test, all_pred_rf))
pipeline_nb = Pipeline([

    ('bow', CountVectorizer(analyzer = text_processing)),

    ('tfidf', TfidfTransformer()),

    ('classifier', MultinomialNB())

    

    

])
pipeline_nb.fit(msg_train, label_train)
all_pred_nb = pipeline_nb.predict(msg_test)
print(classification_report(label_test, all_pred_nb))
from sklearn.linear_model import LogisticRegression
pipeline_lr = Pipeline([

    ('bow', CountVectorizer(analyzer = text_processing)),

    ('tfidf', TfidfTransformer()),

    ('classifier', LogisticRegression())   

])

pipeline_lr.fit(msg_train, label_train)



all_pred_lr = pipeline_lr.predict(msg_test)

print(classification_report(label_test, all_pred_lr))