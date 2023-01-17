import pandas as pd

from pandas import Series,DataFrame

import numpy as np









from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc,precision_score, accuracy_score, recall_score, f1_score

from scipy import interp



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns
#data

df_rd=pd.read_csv('../input/AllTweets.csv')

df_rd.drop(df_rd[df_rd.author=='various'].index, inplace=True)

df_rd.drop(df_rd[df_rd.author=='FiveThirtyEight'].index, inplace=True)
df_rd.drop(df_rd[df_rd.retweet==True].index, inplace=True)
df_rd['length'] = df_rd["text"].apply(len)

df_rd['num_of_words'] = df_rd["text"].str.split().apply(len)
df_rd.hist(column='num_of_words', by='author', bins=100)
df_rd.hist(column='length', by='author', bins=20,figsize=(10,4))
df_rd.drop(df_rd[df_rd.num_of_words<4].index, inplace=True)
df_rd["text"].replace(r"http\S+", "URL", regex=True,inplace=True)
df_rd["text"].replace(r"@\S+", "REF", regex=True ,inplace=True)
df_rd["text"].replace(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+", "DATE", regex=True,inplace=True)
df_rd["text"].replace(r"(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)

df_rd["text"].replace(r"(\d{1,2})[/:](\d{2})?(am|pm)+", "TIME", regex=True,inplace=True)
df_rd["text"].replace(r"\d+", "NUM", regex=True,inplace=True)
df_features=pd.DataFrame()
for a in df_rd.author.unique():

    v = CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 5))

    ngrams = v.fit_transform(df_rd[df_rd['author'] == a]['text'])

    df=pd.DataFrame(

    {'FeaturesNames': v.get_feature_names(),

     'Counts': list(ngrams.sum(axis=0).flat),

     'Author': a

    })

    #

    df['num_of_words'] = df["FeaturesNames"].str.split().apply(len)

    #

    df1=df.loc[(df['num_of_words']==1)].sort_values('Counts', ascending=False)[['Author','Counts','FeaturesNames']].head(100)

    df1.rename(columns={'Counts':'Counts1','FeaturesNames':'Features1'}, inplace=True)

    df1.reset_index(inplace=True)

    #

    df3=df.loc[(df['num_of_words']==3)].sort_values('Counts', ascending=False)[['Counts','FeaturesNames']].head(100)

    df3.rename(columns={'Counts':'Counts3','FeaturesNames':'Features3'}, inplace=True)

    df3.reset_index(inplace=True)

    #

    df5=df.loc[(df['num_of_words']==5)].sort_values('Counts', ascending=False)[['Counts','FeaturesNames']].head(100)

    df5.rename(columns={'Counts':'Counts5','FeaturesNames':'Features5'}, inplace=True)

    df5.reset_index(inplace=True)

    #

    df_result = pd.concat([df1,df3,df5], axis=1)

    #

    df_features=df_features.append(df_result,ignore_index=True)
df_features.drop('index', axis=1, inplace=True)
df_features.loc[(df_features['Author'] == 'NASA')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'NASA'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'AdamSavage')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'AdamSavage'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'BarackObama')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'BarackObama'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'DonaldTrump')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'DonaldTrump'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'HillaryClinton')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'HillaryClinton'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'KimKardashian')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'KimKardashian'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'ScottKelly')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'ScottKelly'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'RichardDawkins')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'RichardDawkins'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head()
df_features.loc[(df_features['Author'] == 'deGrasseTyson')].head(10)
df_features[~df_features.Features1.isin(df_features[df_features['Author'] != 'deGrasseTyson'].Features1)].sort_values('Counts1', ascending=False).ix[:,['Author','Counts1','Features1']].head(5)
def text_process(text):

    """

    Takes in a string of text, then performs the following:

    1. Tokenizes and removes punctuation

    2. Removes  stopwords

    3. Stems

    4. Returns a list of the cleaned text

    """



    # tokenizing

    tokenizer = RegexpTokenizer(r'\w+')

    text_processed=tokenizer.tokenize(text)

    

    # removing any stopwords

    stoplist = stopwords.words('english')

    stoplist.append('twitter')

    stoplist.append('pic')

    stoplist.append('com')

    stoplist.append('net')

    stoplist.append('gov')

    stoplist.append('tv')

    stoplist.append('www')

    stoplist.append('twitter')

    stoplist.append('num')

    stoplist.append('date')

    stoplist.append('time')

    stoplist.append('url')

    stoplist.append('ref')



    stoplist.append('nasa')

    stoplist.append('adam')

    stoplist.append('savage')

    stoplist.append('barack')

    stoplist.append('obama')

    stoplist.append('donald')

    stoplist.append('trump')

    stoplist.append('hillary')

    stoplist.append('clinton')

    stoplist.append('kim')

    stoplist.append('kardashian')

    stoplist.append('kardashian')

    stoplist.append('de')

    stoplist.append('grasse')

    stoplist.append('tyson')

    stoplist.append('scott')

    stoplist.append('kelly')

    stoplist.append('richard')

    stoplist.append('dawkins')

    stoplist.append('adamsavage')

    stoplist.append('barackobama')

    stoplist.append('donaldtrump')

    stoplist.append('hillaryclinton')

    stoplist.append('kimkardashian')

    stoplist.append('degrassetyson')

    stoplist.append('scottkelly')

    stoplist.append('richarddawkins')

    stoplist.append('kourtney')

    text_processed = [word.lower() for word in text_processed if word.lower() not in stoplist]

    

    # steming

    porter_stemmer = PorterStemmer()

    

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    



    return text_processed
df_features=pd.DataFrame()
for a in df_rd.author.unique():

    v = CountVectorizer(tokenizer=text_process,ngram_range=(1, 5))

    ngrams = v.fit_transform(df_rd[df_rd['author'] == a]['text'])

    df=pd.DataFrame(

    {'FeaturesNames': v.get_feature_names(),

     'Counts': list(ngrams.sum(axis=0).flat),

     'Author': a

    })

    #

    df['num_of_words'] = df["FeaturesNames"].str.split().apply(len)

    #

    df1=df.loc[(df['num_of_words']==1)].sort_values('Counts', ascending=False)[['Author','Counts','FeaturesNames']].head()

    df1.rename(columns={'Counts':'Counts1','FeaturesNames':'Features1'}, inplace=True)

    df1.reset_index(inplace=True)

    #

    df3=df.loc[(df['num_of_words']==3)].sort_values('Counts', ascending=False)[['Counts','FeaturesNames']].head()

    df3.rename(columns={'Counts':'Counts3','FeaturesNames':'Features3'}, inplace=True)

    df3.reset_index(inplace=True)

    #

    df5=df.loc[(df['num_of_words']==5)].sort_values('Counts', ascending=False)[['Counts','FeaturesNames']].head()

    df5.rename(columns={'Counts':'Counts5','FeaturesNames':'Features5'}, inplace=True)

    df5.reset_index(inplace=True)

    #

    df_result = pd.concat([df1,df3,df5], axis=1)

    #

    df_features=df_features.append(df_result,ignore_index=True)
df_features.drop('index', axis=1, inplace=True)
df_features.loc[(df_features['Author'] == 'NASA')]
df_features.loc[(df_features['Author'] == 'AdamSavage')]
df_features.loc[(df_features['Author'] == 'BarackObama')]
df_features.loc[(df_features['Author'] == 'DonaldTrump')]
df_features.loc[(df_features['Author'] == 'HillaryClinton')]
df_features.loc[(df_features['Author'] == 'KimKardashian')]
df_features.loc[(df_features['Author'] == 'ScottKelly')]
df_features.loc[(df_features['Author'] == 'RichardDawkins')]
df_features.loc[(df_features['Author'] == 'deGrasseTyson')]
df_features=pd.DataFrame()
for a in df_rd.author.unique():

    v = CountVectorizer(analyzer='char_wb',max_features=2000,ngram_range=(3, 3))

    ngrams = v.fit_transform(df_rd[df_rd['author'] == a]['text'])

    df=pd.DataFrame(

    {'FeaturesNames': v.get_feature_names(),

     'Counts': list(ngrams.sum(axis=0).flat),

     'Author': a

    })

    #

    df_features=df_features.append(df,ignore_index=True)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'NASA'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'AdamSavage'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'BarackObama'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'DonaldTrump'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'HillaryClinton'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'KimKardashian'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'ScottKelly'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'RichardDawkins'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)
df_features[~df_features.FeaturesNames.isin(df_features[df_features['Author'] != 'deGrasseTyson'].FeaturesNames)].sort_values('Counts', ascending=False).head(10)