import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import sklearn
import tensorflow
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test  = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_train.head()
df_test.head()

print(df_train.shape)
df_test.shape
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
import nltk

nltk.download_shell()   

# Stopwords is already installed.
from nltk.corpus import stopwords
print(df_train.columns)
print(df_test.columns)
print(df_train.info())
df_test.info()

print(df_train.isna().sum())

# only 2 null values out of 24000 total values.
# let's drop them.


df_test.isna().sum()

# No null values in test dataset.
df_train.dropna(inplace=True)

df_train.isna().sum()

# No null values left.
# adding a column of text_length
df_train['text_length'] = df_train['text'].apply(lambda x : len(x))

df_test['text_length'] = df_test['text'].apply(lambda x : len(x))

sns.set_style(style='whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(df_train['text_length'],color='green')

# normal distributed data
g = sns.FacetGrid(data=df_train,col='sentiment',height=4)
g.map(sns.distplot,'text_length')
df_train['sentiment'].value_counts().iplot(kind='bar',color='black')

# Maximum Neutral texts
df_test['sentiment'].value_counts().iplot(kind='bar',color='purple')
import string

print(df_train['text'][4])
df_train['selected_text'][4].split()

def sel_tex(i):
    split_text = i.split()
    return split_text
df_train['selected_text2'] = df_train['selected_text'].apply(sel_tex)

df_train.head()
# selected_text column of test dataset will bo on the basis of selected_text of Train dataset to 
#    predict better for types of messages.


select_text = pd.Series(df_train['selected_text'])


list1 = ' '.join(select_text)


list2 = list1.split()
def test_select(i):
    l  = [ ]
    for w in i.split():
        if w in list2:
            l.append(w)
    return(l)
df_test['selected_text'] = df_test['text'].apply(test_select)

df_test.head(6)

df_train.head(1)

from sklearn.feature_extraction.text import CountVectorizer

bag_of_words = CountVectorizer(analyzer=test_select).fit(df_test['text'])
df_test_bow_trans = bag_of_words.transform(df_test['text'])
df_test_bow_trans

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit(df_test_bow_trans)
df_test_tfidf = tfidf.transform(df_test_bow_trans)
df_test_tfidf.shape
from sklearn.naive_bayes import MultinomialNB
sentiment_detect_model = MultinomialNB().fit(df_test_tfidf,df_test['sentiment'])
all_sentiments_predictions = sentiment_detect_model.predict(df_test_tfidf)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(all_sentiments_predictions,df_test['sentiment']))

print(classification_report(all_sentiments_predictions,df_test['sentiment']))


# ACCURACY = 81 %
df_test  = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

df_test.head()

df_test['text_length'] = df_test['text'].apply(lambda x : len(x))

def test_select(i):
    list_text = [text for text in i if text not in string.punctuation]
    join_test_text = ''.join(list_text)
    clean_test_text = [ text for text in join_test_text.split() if text.lower() not in stopwords.words('english')]
    return clean_test_text
df_test['selected_text'] = df_test['text'].apply(test_select)

df_test.head()

bag_of_words = CountVectorizer(analyzer=test_select).fit(df_test['text'])


df_test_bow_trans = bag_of_words.transform(df_test['text'])


tfidf = TfidfTransformer().fit(df_test_bow_trans)


df_test_tfidf = tfidf.transform(df_test_bow_trans)


sentiment_detect_model = MultinomialNB().fit(df_test_tfidf,df_test['sentiment'])


all_sentiments_predictions = sentiment_detect_model.predict(df_test_tfidf)
print(confusion_matrix(all_sentiments_predictions,df_test['sentiment']))

print(classification_report(all_sentiments_predictions,df_test['sentiment']))


# ACCURACY = 91 %
# Therefore , option 2 has increased accuracy by 10%.
df_test.head(2)
def joined(i):
    joined = " , ".join(i)
    return joined
df_test['selected_text2'] = df_test['selected_text'].apply(joined)
df_test.head()
df_test2 = df_test[['textID','selected_text2']]
df_test2.rename(columns={'selected_text2':'selected_text'},inplace=True)
df_test2.head(1)
df_test2.to_csv('submission.csv',index=False)
