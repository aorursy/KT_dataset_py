import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from nltk.stem.wordnet import WordNetLemmatizer

stop_words =set( stopwords.words('english') )

import string

punc=set(string.punctuation)

lemm=WordNetLemmatizer()

lemmatizer = WordNetLemmatizer()
df=pd.read_csv("../input/urban_dictionary.csv")
df.head()
df.shape
df.info()
df.drop(['date','tags'],axis=1,inplace=True)
df.dtypes
df['author'].value_counts().head()
df['word'].value_counts().head()
df.columns
df['original_count']=df.definition.apply(lambda x:len(x.split()))
df.head()
print('positive word',df.loc[df.up==df.up.max()]['word'])

print('Negative  word',df.loc[df.down==df.down.max()]['word'])
df['def_new']=None

for i in range(len(df['definition'])):

    doc=df.definition[i]

    doc=doc.split(" ")

    doc=[w for w in doc if w not in set(stop_words)]

    doc=[w for w in doc if w not in punc]

    doc=" ".join([lemmatizer.lemmatize(word) for word in doc])

    df.at[i,'def_new']=doc



    
df.head()
df['def_new_count']=df['def_new'].apply(lambda x:len(x.split(" ")))
df[['def_new_count','original_count']].head()
sm=SentimentIntensityAnalyzer()
df['score']=None

df['polarity']=None

for i in range(len(df.def_new)):

    score_dic=sm.polarity_scores(df.def_new[i])

    key=max(score_dic,key=score_dic.get)

    df.at[i,'score']=score_dic[key]

    df.at[i,'polarity']=key

    #print(key,score_dic[key])

    

    
val=(df['polarity'].value_counts())
type(val)
df['polarity'].value_counts().index
pd.DataFrame(val).plot.pie(y='polarity',figsize=(10,10),autopct='%1.0f%%')