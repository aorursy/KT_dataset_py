import pandas as pd

import numpy as np

import nltk as nltk

import sklearn as sklearn

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv(r"D:\CourseWork\DataScience\CORD-19-research-challenge\all_sources_metadata_2020-03-13.csv")
type(df)
df.shape
df.head(3)
df.isnull().sum()
df.drop(['sha','Microsoft Academic Paper ID','doi','WHO #Covidence','has_full_text','pmcid','pubmed_id','license','authors'],axis=1,inplace=True)
df.isnull().sum()
df.dropna(subset=['title','abstract','publish_time','journal'],inplace=True)

df.isnull().sum()
df.shape
df.head(3)
cols=['title','abstract','journal','source_x']

for cols in cols:

    df[cols]=df[cols].str.lower()
#df[cols]=df[cols].apply(lambda x:x.str.lower())
df.head(3)
cols=['title','abstract','journal','source_x']

for cols in cols:

    df[cols] = df[cols].str.replace('[^\w\s\d]','')
df.head(3)
from nltk.corpus import stopwords

stop = stopwords.words('english')

cols=['title','abstract','journal','source_x']

df[cols]=df[cols].apply(lambda i:" ".join([word for word in i.split() if word not in stop]))
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

cols=['title','abstract','journal','source_x']

df[cols] = df[cols].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))



    
df.head(3)
df.shape
Corpus=" ".join(df['title'])



def word_counter(YourText):

    word_counter={}

    for word in YourText.split():

        if word not in word_counter:

            word_counter[word]=1

        else:

            word_counter[word]+=1

    return word_counter



word_freq=word_counter(Corpus)

word_counter(Corpus)



df_word_freq=pd.DataFrame({'word':list(word_freq.keys()), 'freq':list(word_freq.values())})

df_word_freq.sort_values(['freq'],ascending=True,inplace=True)

rare_words=list(df_word_freq.word[0:10])

df_word_freq.sort_values(['freq'],ascending=False,inplace=True)

df_word_freq
most_common_words=df_word_freq[0:20]

most_common_words.reset_index(inplace=True)

most_common_words.drop('index',axis=1,inplace=True)

most_common_words
sns.barplot(most_common_words.freq,most_common_words.word).set_title('Most Used Words in title of the dataset')
from wordcloud import WordCloud



bowcloud= WordCloud(background_color='White').generate(str(most_common_words))

plt.imshow(bowcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = df['title']



vect=TfidfVectorizer(stop_words='english',sublinear_tf=True, max_df=0.5, analyzer='word',max_features=100,ngram_range=(1,2))

X=vect.fit_transform(corpus)

colnames=vect.get_feature_names()

df[colnames]=pd.DataFrame(X.toarray(), columns=colnames)



df.head(5)
Corpus=" ".join(df['abstract'])

word_freq=word_counter(Corpus)

word_counter(Corpus)

df_word_freq=pd.DataFrame({'word':list(word_freq.keys()), 'freq':list(word_freq.values())})

df_word_freq.sort_values(['freq'],ascending=True,inplace=True)

rare_words=list(df_word_freq.word[0:10])

df_word_freq.sort_values(['freq'],ascending=False,inplace=True)

common_words=list(df_word_freq.word[0:10])
most_common_words2=df_word_freq[0:20]

most_common_words2.reset_index(inplace=True)

most_common_words2.drop('index',axis=1,inplace=True)

most_common_words2
sns.barplot(most_common_words2.freq,most_common_words2.word).set_title('Most Used Words in abstract of the dataset')
from wordcloud import WordCloud



bowcloud= WordCloud(background_color='White').generate(str(most_common_words2))

plt.imshow(bowcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

corpus1 = df['abstract']



vect=TfidfVectorizer(stop_words='english',sublinear_tf=True, max_df=0.5, analyzer='word',max_features=100,ngram_range=(1,2))

Y=vect.fit_transform(corpus1)

colnames1=vect.get_feature_names()

df[colnames1]=pd.DataFrame(Y.toarray(), columns=colnames1)

df.head(5)
df.shape
Corpus=" ".join(df['journal'])



word_freq=word_counter(Corpus)

word_counter(Corpus)

df_word_freq=pd.DataFrame({'word':list(word_freq.keys()), 'freq':list(word_freq.values())})

df_word_freq.sort_values(['freq'],ascending=True,inplace=True)

rare_words=list(df_word_freq.word[0:10])

df_word_freq.sort_values(['freq'],ascending=False,inplace=True)

common_words=list(df_word_freq.word[0:10])
most_common_words3=df_word_freq[20:40]

most_common_words3.reset_index(inplace=True)

most_common_words3.drop('index',axis=1,inplace=True)

most_common_words3
sns.barplot(most_common_words3.freq,most_common_words3.word).set_title('Most Used Words in journal of the dataset')
from wordcloud import WordCloud



bowcloud= WordCloud(background_color='White').generate(str(most_common_words3))

plt.imshow(bowcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

corpus2 = df['journal']



vect=TfidfVectorizer(stop_words='english',sublinear_tf=True, max_df=0.5, analyzer='word',max_features=100,ngram_range=(1,2))

Z=vect.fit_transform(corpus2)

colnames2=vect.get_feature_names()

df[colnames2]=pd.DataFrame(Y.toarray(), columns=colnames2)

df.head(5)
df.drop(['title','abstract'],axis=1,inplace=True)
df.head(3)
df.describe()
df.head(3)
df.drop(['source_x','publish_time'],axis=1,inplace=True)
df.head(3)
BOW=df.columns
from wordcloud import WordCloud



bowcloud= WordCloud(background_color='White').generate(str(BOW))

plt.imshow(bowcloud,interpolation='bilinear')

plt.axis('off')

plt.show()