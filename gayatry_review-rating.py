import pandas as pd

import numpy as np

from pandas import Series,DataFrame



import string

from nltk.corpus import stopwords



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/shopee-code-league-20/_DS_Sentiment_Analysis/train.csv',names=['message','rating'])

df = df.iloc[1:]
df.head()
df.info()
df.describe()
df.groupby('rating').describe()
df['length'] = df['message'].apply(len)

df.head()
df['length'].plot(bins=50,kind='hist')
df['length'].describe()
df[df['length']==1249]['message'].iloc[0]
df.hist(column='length',by='rating',bins=50,figsize=(10,8))
def textprocess(mess):

    """Removing punctuation """

    nonpunc = [char for char in mess if char not in string.punctuation]

    nonpunc = ''.join(nonpunc)

    

    """Removing stopwords"""

    clean_mess = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]

    

    return clean_mess
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=textprocess).fit(df['message'])
print(len(bow_transformer.vocabulary_))
messages_bow = bow_transformer.transform(df['message'])
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)

messages_tfidf = tfidf_transformer.transform(messages_bow)
from sklearn.naive_bayes import MultinomialNB
senti_analysis = MultinomialNB().fit(messages_tfidf,df['rating'])
all_predictions = senti_analysis.predict(messages_tfidf)
d = {'Predicted':all_predictions,'Actual':df['rating']}

df_analysis = DataFrame(d)
from sklearn.metrics import classification_report
print(classification_report(df['rating'],all_predictions))
from sklearn.model_selection import train_test_split



msg_train,msg_test,rating_train,rating_test = train_test_split(df['message'],df['rating'])



print(len(msg_train), len(msg_test), len(msg_train),len(msg_test))
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=textprocess)),

    ('tfidf', TfidfTransformer()),

    ('classifier', MultinomialNB())

])
pipeline.fit(msg_train,rating_train)
pred_rate = pipeline.predict(msg_test)
print(classification_report(rating_test,pred_rate))
df1 = pd.read_csv('../input/shopee-code-league-20/_DS_Sentiment_Analysis/test.csv',index_col=['review_id'])

df1 = df1.rename(columns={'review':'message'})
prediction = pipeline.predict(df1['message'])
df1['Rating'] = prediction

df1