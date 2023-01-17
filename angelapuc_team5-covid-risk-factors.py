import pandas as pd

import numpy as np

import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, stem_text

from nltk.corpus import stopwords

import pickle

import en_core_web_sm

import csv

import json

import nltk

import langid

from gensim.test.utils import common_texts, get_tmpfile

from gensim.models import Word2Vec

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from scipy.spatial.distance import cosine, cdist

import matplotlib.pyplot as plt
path = '../input/CORD-19-research-challenge/'

path2 = '../input/CORD-19-research-challenge/Kaggle/target_tables/8_risk_factors/'
#loading creating the dataframes

meta_df = pd.read_csv(path + 'metadata.csv') 

meta_df = meta_df[['publish_time','title','abstract','cord_uid', 'doi', 'journal', 'url','pdf_json_files']]



target_smoking_df = pd.read_csv(path2 +'Smoking Status.csv', index_col= 'Unnamed: 0')

target_smoking_df = target_smoking_df[['Date', 'Study', 'Study Link', 'Journal', 'Study Type','Added on']]

target_diabetes_df = pd.read_csv(path2 +'Diabetes.csv', index_col= 'Unnamed: 0')

target_diabetes_df = target_diabetes_df[['Date', 'Study', 'Study Link', 'Journal', 'Study Type','Added on']]

target_hypertension_df = pd.read_csv(path2+'Hypertension.csv', index_col= 'Unnamed: 0')

target_hypertension_df = target_hypertension_df[['Date', 'Study', 'Study Link', 'Journal', 'Study Type','Added on']]

print('meta length:',len(meta_df))

print('smoking target table length:',len(target_smoking_df))

print('diabetes target table length:',len(target_diabetes_df))

print('hypertension target table length:',len(target_hypertension_df))
#Merging the abstracts in the target tables

target_smoking_df = target_smoking_df.merge(meta_df,how='inner', left_on='Study', right_on='title')

#target_smoking_df = target_smoking_df[['Date','title','abstract']]

target_diabetes_df = target_diabetes_df.merge(meta_df,how='inner', left_on='Study', right_on='title')

#target_diabetes_df = target_diabetes_df[['Date','title','abstract']] >meta_df[['title', 'abstract']]

target_hypertension_df = target_hypertension_df.merge(meta_df,how='inner', left_on='Study', right_on='title')

#target_hypertension_df = target_hypertension_df[['Date','title','abstract']]
plt.figure(figsize=(20,10))

meta_df.isna().sum().plot(kind='bar', stacked=True)
#Getting rid of duplicates and Null Values in meta dataframe

print("Initial length:",len(meta_df))

meta_df.drop_duplicates(subset='title', keep="first", inplace=True)

meta_df.drop_duplicates(subset='abstract', keep="first", inplace=True)

print("After dropping duplicates:",len(meta_df))

meta_df.dropna(axis=0, inplace=True, subset=['publish_time','title','abstract'])

print("After dropping N/A:",len(meta_df))

meta_df.reset_index(inplace=True)
#Getting rid of duplicates and Null Values in target dataframes

print("Initial length-somking:",len(target_smoking_df))

target_smoking_df.drop_duplicates(subset='title', keep="first", inplace=True)

target_smoking_df.drop_duplicates(subset='abstract', keep="first", inplace=True)

print("After dropping duplicates-somking:",len(target_smoking_df))

target_smoking_df.dropna(axis=0, inplace=True, subset = ['title','abstract'])

print("After dropping N/A-somking:",len(target_smoking_df))

target_smoking_df.reset_index(inplace=True)

print("Initial length-diabetes:",len(target_diabetes_df))

target_diabetes_df.drop_duplicates(subset='title', keep="first", inplace=True)

target_diabetes_df.drop_duplicates(subset='abstract', keep="first", inplace=True)

print("After dropping duplicates-diabetes:",len(target_diabetes_df))

target_diabetes_df.dropna(axis=0, inplace=True, subset = ['title','abstract'])

print("After dropping N/A-diabetes:",len(target_diabetes_df))

target_diabetes_df.reset_index(inplace=True)

print("Initial length-hypertension:",len(target_hypertension_df))

target_hypertension_df.drop_duplicates(subset='title', keep="first", inplace=True)

target_hypertension_df.drop_duplicates(subset='abstract', keep="first", inplace=True)

print("After dropping duplicates-hypertension:",len(target_hypertension_df))

target_hypertension_df.dropna(axis=0, inplace=True, subset = ['title','abstract'])

print("After dropping N/A-hypertension:",len(target_hypertension_df))

target_hypertension_df.reset_index(inplace=True)
target_smoking_df.head(6)
target_diabetes_df.head(6)
target_hypertension_df.head(6)
#Creating the year column

meta_df['date'] = pd.to_datetime(meta_df['publish_time'], format ='%Y-%m-%d',errors='coerce')

meta_df['year'] = pd.DatetimeIndex(meta_df['date']).year.fillna(0).astype(int)

meta_df.head()
#Filtering our metadata based on publish year

meta_df = meta_df[meta_df['year']>2019]

meta_df.reset_index(inplace=True, drop=True)

print('After publish year filter',len(meta_df))

meta_df.head()
for i in range(len(meta_df.abstract)):

        meta_df.abstract[i] =  strip_numeric(meta_df.abstract[i]) #Remove digits

        meta_df.abstract[i] =  strip_punctuation(str(meta_df.abstract[i]))  #Remove punctuation

        meta_df.abstract[i] =  strip_multiple_whitespaces(str(meta_df.abstract[i])) #Remove multiple whitespaces
for i in range(len(target_smoking_df.abstract)):

        target_smoking_df.abstract[i] =  strip_numeric(target_smoking_df.abstract[i]) #Remove digits

        target_smoking_df.abstract[i] =  strip_punctuation(str(target_smoking_df.abstract[i]))  #Remove punctuation

        target_smoking_df.abstract[i] =  strip_multiple_whitespaces(str(target_smoking_df.abstract[i])) #Remove multiple whitespaces
for i in range(len(target_diabetes_df.abstract)):

        target_diabetes_df.abstract[i] =  strip_numeric(target_diabetes_df.abstract[i]) #Remove digits

        target_diabetes_df.abstract[i] =  strip_punctuation(str(target_diabetes_df.abstract[i]))  #Remove punctuation

        target_diabetes_df.abstract[i] =  strip_multiple_whitespaces(str(target_diabetes_df.abstract[i])) #Remove multiple whitespaces
for i in range(len(target_hypertension_df.abstract)):

        target_hypertension_df.abstract[i] =  strip_numeric(target_hypertension_df.abstract[i]) #Remove digits

        target_hypertension_df.abstract[i] =  strip_punctuation(str(target_hypertension_df.abstract[i]))  #Remove punctuation

        target_hypertension_df.abstract[i] =  strip_multiple_whitespaces(str(target_hypertension_df.abstract[i])) #Remove multiple whitespaces
#Turning everything lowecase

for i in range(len(meta_df.abstract)):

    meta_df.abstract[i] = meta_df.abstract[i].lower()

for i in range(len(target_smoking_df.abstract)):

    target_smoking_df.abstract[i] = target_smoking_df.abstract[i].lower()

for i in range(len(target_diabetes_df.abstract)):

    target_diabetes_df.abstract[i] = target_diabetes_df.abstract[i].lower()

for i in range(len(target_hypertension_df.abstract)):

    target_hypertension_df.abstract[i] = target_hypertension_df.abstract[i].lower()
#making a back-up just in case :)

backup_meta=meta_df

backup_target_smoking=target_smoking_df

backup_target_diabetes=target_diabetes_df

backup_meta_target_hypertension=target_hypertension_df
# filtering for covid Literature (meta df)

searchfor = ['covid','corona','ncov']

meta_df = meta_df[meta_df['abstract'].str.contains('|'.join(searchfor), na = False, case=False)] #doesn't consider NA and is case insensitive

meta_df.reset_index(inplace=True, drop=True)

print('After applying covid lit.', len(meta_df))

meta_df
#Assigning languages to each article and filtering on English (meta df)

meta_df['language']='unknown'

for i in range(len(meta_df['abstract'])):   

    meta_df['language'][i]=langid.classify(meta_df['abstract'][i])[0]

meta_df=meta_df[meta_df.language.isin(['en'])]

meta_df.reset_index(inplace=True, drop=True)

print('After applying languege filter', len(meta_df))

meta_df.head()
#removing the stopwords and short words (less than 3)

for i in range(len(meta_df.abstract)):

    meta_df.abstract[i] = remove_stopwords(meta_df.abstract[i])

    meta_df.abstract[i] = strip_short(meta_df.abstract[i])

    

for i in range(len(target_smoking_df.abstract)):

    target_smoking_df.abstract[i] = remove_stopwords(target_smoking_df.abstract[i])

    target_smoking_df.abstract[i] = strip_short(target_smoking_df.abstract[i])

    

for i in range(len(target_diabetes_df.abstract)):

    target_diabetes_df.abstract[i] = remove_stopwords(target_diabetes_df.abstract[i])

    target_diabetes_df.abstract[i] = strip_short(target_diabetes_df.abstract[i])

    

for i in range(len(target_hypertension_df.abstract)):

    target_hypertension_df.abstract[i] = remove_stopwords(target_hypertension_df.abstract[i])

    target_hypertension_df.abstract[i] = strip_short(target_hypertension_df.abstract[i])
#making copies for stemming 

stemmed_meta = meta_df.copy()



stemmed_smoking = target_smoking_df.copy()

#stemmed_smoking = stemmed_smoking.drop(columns=['index'])



stemmed_hypertension = target_hypertension_df.copy()

#stemmed_hypertension = stemmed_hypertension.drop(columns=['index'])



stemmed_diabetes = target_diabetes_df.copy()

#stemmed_diabetes = stemmed_diabetes.drop(columns=['index'])
#Stemming

for i in range(len(stemmed_meta.abstract)):

    stemmed_meta.abstract[i] = stem_text(stemmed_meta.abstract[i])



for i in range(len(stemmed_smoking.abstract)):

    stemmed_smoking.abstract[i] = stem_text(stemmed_smoking.abstract[i])



for i in range(len(stemmed_diabetes.abstract)):

    stemmed_diabetes.abstract[i] = stem_text(stemmed_diabetes.abstract[i])



for i in range(len(stemmed_hypertension.abstract)):

    stemmed_hypertension.abstract[i] = stem_text(stemmed_hypertension.abstract[i])
print(stemmed_meta.abstract[0])

print('')

print(stemmed_smoking.abstract[0])

print('')

print(stemmed_diabetes.abstract[0])

print('')

print(stemmed_hypertension.abstract[0])
#Making a Matrix representing the count of each word in meta

vectorizer_meta = CountVectorizer(max_features = 100)

X_meta = vectorizer_meta.fit_transform(stemmed_meta.abstract)



#Creating count of BOW for all articles in meta

corpus_meta = vectorizer_meta.get_feature_names()

corpus_meta = np.asarray(corpus_meta) #converting corpus to np.array

count_meta = pd.DataFrame(data=corpus_meta) #creating the data frame

count_meta.rename(columns={0 :'Key'}, inplace=True)

count_meta.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_meta.abstract)):

    count_meta[str(i)] = X_meta[i].toarray()[0]
vectorizer_smoking = CountVectorizer(min_df=0.05)

X_smoking = vectorizer_smoking.fit_transform(stemmed_smoking.abstract)



#Creating count of BOW for all articles in target

corpus_smoking = vectorizer_smoking.get_feature_names()

corpus_smoking = np.asarray(corpus_smoking) #converting corpus to np.array

count_smoking = pd.DataFrame(data=corpus_smoking) #creating the data frame

count_smoking.rename(columns={0 :'Key'}, inplace=True)

count_smoking.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_smoking.abstract)):

    count_smoking[str(i)] = X_smoking[i].toarray()[0]
vectorizer_diabetes = CountVectorizer(min_df=0.05)

X_diabetes = vectorizer_diabetes.fit_transform(stemmed_diabetes.abstract)

    

corpus_diabetes = vectorizer_diabetes.get_feature_names()

corpus_diabetes = np.asarray(corpus_diabetes) #converting corpus to np.array

count_diabetes = pd.DataFrame(data=corpus_diabetes) #creating the data frame

count_diabetes.rename(columns={0 :'Key'}, inplace=True)

count_diabetes.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_diabetes.abstract)):

    count_diabetes[str(i)] = X_diabetes[i].toarray()[0]
vectorizer_hypertension = CountVectorizer(min_df=0.05)

X_hypertension = vectorizer_hypertension.fit_transform(stemmed_hypertension.abstract)



corpus_hypertension = vectorizer_hypertension.get_feature_names()

corpus_hypertension = np.asarray(corpus_hypertension) #converting corpus to np.array

count_hypertension = pd.DataFrame(data=corpus_hypertension) #creating the data frame

count_hypertension.rename(columns={0 :'Key'}, inplace=True)

count_hypertension.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_hypertension.abstract)):

    count_hypertension[str(i)] = X_hypertension[i].toarray()[0]
#Adding a total column

count_meta.loc[:,'Total'] = count_meta.sum(numeric_only=True, axis=1)

count_smoking.loc[:,'Total'] = count_smoking.sum(numeric_only=True, axis=1)

count_diabetes.loc[:,'Total'] = count_diabetes.sum(numeric_only=True, axis=1)

count_hypertension.loc[:,'Total'] = count_hypertension.sum(numeric_only=True, axis=1)
#sorting based on total counts

count_meta = count_meta.sort_values('Total', ascending = False)

count_smoking = count_smoking.sort_values('Total', ascending = False)

count_diabetes = count_diabetes.sort_values('Total', ascending = False)

count_hypertension = count_hypertension.sort_values('Total', ascending = False)
count_meta
count_smoking
count_diabetes
count_hypertension
#making smaller dfs

count_meta = count_meta['Total']

count_smoking = count_smoking['Total']

count_diabetes = count_diabetes['Total']

count_hypertension = count_hypertension['Total']   
count_meta.head()
count_smoking.head()
count_diabetes.head()
count_hypertension.head()
#word cloud to see what our meta data is mostly about

%matplotlib inline

from wordcloud import WordCloud, STOPWORDS

text = meta_df.abstract

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
print("Smoking word cloud:")

text = target_smoking_df.abstract

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
print ("Diabetes word cloud:")

text = target_diabetes_df.abstract

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
print ("Hypertension word cloud:")

text = target_hypertension_df.abstract

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#top 100 words in meta we will choose the noise from them

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(count_meta.index.tolist())
noise = ['re', 'charactist', 'dim','chronic' , 'obsv', 'intub','mortal', 'sev', 'charactist', 'obsv','conduct','admiss', 'factor', 'death', 'sex', 'male', 'old', 'march', 'blood', 'great', 'low', 'discharg', 'adult', 'bmi', 'logist', 'diff', 'respect', 'icu', 'fatal', 'main', 'numb', 'point', 'state', 'er', 'ratio','independ','characterist','critic','predict','progress','meta','odd','multivari','ill','non','laboratori','nlr','cohort','count', 'record', 'find', 'regress','median', 'score','observ', 'retrospect','survivor', 'assess','significantli', 'adjt', 'total','earli','index','help', 'analys', 'unit','admit', 'preval', 'peopl','analyz','common', 'scienc','februari','follow', 'signific','calcul', 'collect','like', 'evalu', 'examin', 'search','random','end','januari','suggest','background', 'articl', 'copyright', 'rightreserv', 'object', 'import', 'aim', 'covid','patient','cov','infect','diseas','sar','case','coronaviru','pandem','studi','health','sever','result','clinic','risk','data','respiratori','test','hospit','care','report','includ','time','effect','model','treatment','method','viru','outbreak','number','dai','increas','provid','spread','base','associ','acut','measur','develop','control','symptom','china','caus','rate','differ','countri','present','epidem','high','conclus','emerg','current','group','us','public','syndrom','posit','viral','need','level','confirm','transmiss','medic','novel','respons','popul','new','potenti','ag','prevent','identifi','analysi','compar','cell','manag','human','global','review','social','outcom','relat','world','year','drug','detect','import','impact','inform','protect','specif','activ','perform','higher','gener','wuhan','avail','show']

print(len(noise))
#removing noise words from meta

#noise = ['re', 'charactist', 'dim','chronic' , 'obsv', 'intub','mortal', 'sev', 'charactist', 'obsv','conduct','admiss', 'factor', 'death', 'sex', 'male', 'old', 'march', 'blood', 'great', 'low', 'discharg', 'adult', 'bmi', 'logist', 'diff', 'respect', 'icu', 'fatal', 'main', 'numb', 'point', 'state', 'er', 'ratio','independ','characterist','critic','predict','progress','meta','odd','multivari','ill','non','laboratori','nlr','cohort','count', 'record', 'find', 'regress','median', 'score','observ', 'retrospect','survivor', 'assess','significantli', 'adjt', 'total','earli','index','help', 'analys', 'unit','admit', 'preval', 'peopl','analyz','common', 'scienc','februari','follow', 'signific','calcul', 'collect','like', 'evalu', 'examin', 'search','random','end','januari','suggest','background', 'articl', 'copyright', 'rightreserv', 'object', 'import', 'aim', 'covid','patient','cov','infect','diseas','sar','case','coronaviru','pandem','studi','health','sever','result','clinic','risk','data','respiratori','test','hospit','care','report','includ','time','effect','model','treatment','method','viru','outbreak','number','dai','increas','provid','spread','base','associ','acut','measur','develop','control','symptom','china','caus','rate','differ','countri','present','epidem','high','conclus','emerg','current','group','us','public','syndrom','posit','viral','need','level','confirm','transmiss','medic','novel','respons','popul','new','potenti','ag','prevent','identifi','analysi','compar','cell','manag','human','global','review','social','outcom','relat','world','year','drug','detect','import','impact','inform','protect','specif','activ','perform','higher','gener','wuhan','avail','show']

for i in range(len(stemmed_meta['abstract'])):

    for j in range(len(noise)):

        stemmed_meta['abstract'][i] = re.sub(noise[j], r'', stemmed_meta['abstract'][i]) 
#removing noise words from target tables

for i in range(len(stemmed_smoking['abstract'])):

    for j in range(len(noise)):

        stemmed_smoking['abstract'][i] = re.sub(noise[j], r'', stemmed_smoking['abstract'][i]) 



for i in range(len(stemmed_diabetes['abstract'])):

    for j in range(len(noise)):

        stemmed_diabetes['abstract'][i] = re.sub(noise[j], r'', stemmed_diabetes['abstract'][i]) 



for i in range(len(stemmed_hypertension['abstract'])):

    for j in range(len(noise)):

        stemmed_hypertension['abstract'][i] = re.sub(noise[j], r'', stemmed_hypertension['abstract'][i]) 
#Applying the strip short and multiple white spaces on all documents again

for i in range(len(stemmed_meta.abstract)):

    stemmed_meta.abstract[i] = strip_short(stemmed_meta.abstract[i], minsize = 2)

    stemmed_meta.abstract[i] = strip_multiple_whitespaces(str(stemmed_meta.abstract[i]))



for i in range(len(stemmed_smoking.abstract)):

    stemmed_smoking.abstract[i] = strip_short(stemmed_smoking.abstract[i], minsize = 2)

    stemmed_smoking.abstract[i] = strip_multiple_whitespaces(str(stemmed_smoking.abstract[i]))



for i in range(len(stemmed_diabetes.abstract)):

    stemmed_diabetes.abstract[i] = strip_short(stemmed_diabetes.abstract[i], minsize = 2)

    stemmed_diabetes.abstract[i] = strip_multiple_whitespaces(str(stemmed_diabetes.abstract[i]))



for i in range(len(stemmed_hypertension.abstract)):

    stemmed_hypertension.abstract[i] = strip_short(stemmed_hypertension.abstract[i], minsize = 2)

    stemmed_hypertension.abstract[i] = strip_multiple_whitespaces(str(stemmed_hypertension.abstract[i]))



print(stemmed_meta['abstract'][0])

print('')

print(stemmed_smoking['abstract'][0])

print('')

print(stemmed_diabetes['abstract'][0])

print('')

print(stemmed_hypertension['abstract'][0])  
#We recreate the count dataframe for the target table to see the effects ==> smoking 

vectorizer_smoking = CountVectorizer(min_df=0.05)

X_smoking = vectorizer_smoking.fit_transform(stemmed_smoking.abstract)

#Creating count of BOW for all articles in target

corpus_smoking = vectorizer_smoking.get_feature_names()

corpus_smoking = np.asarray(corpus_smoking) #converting corpus to np.array

count_smoking = pd.DataFrame(data=corpus_smoking) #creating the data frame

count_smoking.rename(columns={0 :'Key'}, inplace=True)

count_smoking.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_smoking.abstract)):

    count_smoking[str(i)] = X_smoking[i].toarray()[0]

count_smoking.loc[:,'Total'] = count_smoking.sum(numeric_only=True, axis=1)

count_smoking = count_smoking.sort_values('Total', ascending = False)

count_smoking = count_smoking['Total']

print ("New smoking word cloud:")

d = {}

d = count_smoking.to_dict()

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS)

wordcloud.generate_from_frequencies(d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#We recreate the count dataframe for the target table to see the effects ==> diabetes

vectorizer_diabetes = CountVectorizer(min_df=0.05)

X_diabetes = vectorizer_diabetes.fit_transform(stemmed_diabetes.abstract)

#Creating count of BOW for all articles in target

corpus_diabetes = vectorizer_diabetes.get_feature_names()

corpus_diabetes = np.asarray(corpus_diabetes) #converting corpus to np.array

count_diabetes = pd.DataFrame(data=corpus_diabetes) #creating the data frame

count_diabetes.rename(columns={0 :'Key'}, inplace=True)

count_diabetes.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_diabetes.abstract)):

    count_diabetes[str(i)] = X_diabetes[i].toarray()[0]

count_diabetes.loc[:,'Total'] = count_diabetes.sum(numeric_only=True, axis=1)

count_diabetes = count_diabetes.sort_values('Total', ascending = False)

count_diabetes = count_diabetes['Total']

print ("New diabetes word cloud:")

d = {}

d = count_diabetes.to_dict()

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS)

wordcloud.generate_from_frequencies(d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#We recreate the count dataframe for the target table to see the effects ==> hypertension 

vectorizer_hypertension = CountVectorizer(min_df=0.05)

X_hypertension = vectorizer_hypertension.fit_transform(stemmed_hypertension.abstract)

#Creating count of BOW for all articles in target

corpus_hypertension = vectorizer_hypertension.get_feature_names()

corpus_hypertension = np.asarray(corpus_hypertension) #converting corpus to np.array

count_hypertension = pd.DataFrame(data=corpus_hypertension) #creating the data frame

count_hypertension.rename(columns={0 :'Key'}, inplace=True)

count_hypertension.set_index('Key', inplace=True) #setting index as key

#adding the rest of the data

for i in range(len(stemmed_hypertension.abstract)):

    count_hypertension[str(i)] = X_hypertension[i].toarray()[0]

count_hypertension.loc[:,'Total'] = count_hypertension.sum(numeric_only=True, axis=1)

count_hypertension = count_hypertension.sort_values('Total', ascending = False)

count_hypertension = count_hypertension['Total']

print ("New hypertension word cloud:")

d = {}

d = count_hypertension.to_dict()

wordcloud = WordCloud(

    width = 2000,

    height = 1000,

    background_color = 'black',

    stopwords = STOPWORDS)

wordcloud.generate_from_frequencies(d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#dropping duplicates again

#stemmed_meta=stemmed_meta.drop(index=20963)

#stemmed_meta.sort_values('abstract')

#stemmed_meta2 = stemmed_meta.copy()

#for i in range(len(stemmed_meta.abstract)):

#    stemmed_meta2.title[i] = stem_text(stemmed_meta2.title[i])

stemmed_meta.drop_duplicates(subset='title', keep="first", inplace=True)

stemmed_meta.drop_duplicates(subset='abstract', keep="first", inplace=True)

stemmed_meta.reset_index(inplace=True, drop=True)

stemmed_smoking.drop_duplicates(subset='title', keep="first", inplace=True)

stemmed_smoking.drop_duplicates(subset='abstract', keep="first", inplace=True)

stemmed_smoking.reset_index(inplace=True, drop=True)

stemmed_diabetes.drop_duplicates(subset='title', keep="first", inplace=True)

stemmed_diabetes.drop_duplicates(subset='abstract', keep="first", inplace=True)

stemmed_diabetes.reset_index(inplace=True, drop=True)

stemmed_hypertension.drop_duplicates(subset='title', keep="first", inplace=True)

stemmed_hypertension.drop_duplicates(subset='abstract', keep="first", inplace=True)

stemmed_hypertension.reset_index(inplace=True, drop=True)

#print(stemmed_meta.loc[12374,'abstract'])

#print(stemmed_meta.loc[1402, 'abstract'])
filename = 'meta_stemmed'

outfile = open(filename,'wb')

pickle.dump(stemmed_meta,outfile)

outfile.close()
filename = 'smoking_stemmed'

outfile = open(filename,'wb')

pickle.dump(stemmed_smoking,outfile)

outfile.close()
filename = 'diabetes_stemmed'

outfile = open(filename,'wb')

pickle.dump(stemmed_diabetes,outfile)

outfile.close()
filename = 'hypertension_stemmed'

outfile = open(filename,'wb')

pickle.dump(stemmed_hypertension,outfile)

outfile.close()
#loading the stemmed data

#path3 = '../input/output/'

path3 = '../input/stemmed-data/'

#infile = open(path3+'meta_stemmed','rb')

infile = open(path3 +'meta_stemmed','rb')

stemmed_meta = pickle.load(infile)

infile.close()



#loading the stemmed data

infile = open(path3+'smoking_stemmed','rb')

stemmed_smoking = pickle.load(infile)

infile.close()
#loading the stemmed data

infile = open(path3+'hypertension_stemmed','rb')

stemmed_hypertension = pickle.load(infile)

infile.close()
#loading the stemmed data

infile = open(path3+'diabetes_stemmed','rb')

stemmed_diabetes = pickle.load(infile)

infile.close()
# Making the stemmed abstracts into a list

meta_list = list(stemmed_meta.abstract)

smoking_list = list(stemmed_smoking.abstract)

diabetes_list = list(stemmed_diabetes.abstract)

hypertension_list = list(stemmed_hypertension.abstract)
# splitting corpus

meta_corpus = [doc.split() for doc in meta_list]

print(meta_corpus[0])

print('')

smoking_corpus = [doc.split() for doc in smoking_list]

print(smoking_corpus[0])

print('')

diabetes_corpus = [doc.split() for doc in diabetes_list]

print(diabetes_corpus[0])

print('')

hypertension_corpus = [doc.split() for doc in hypertension_list]

print(hypertension_corpus[0])
# initiaize the model building vocabulary with the abstracts from meta

meta_sentences = [TaggedDocument(doc, [i]) for i, doc in enumerate(meta_corpus)]

d2v_model = Doc2Vec(vector_size=20, min_count=5, workers=11, alpha=0.025, epochs=200)

d2v_model.build_vocab(meta_sentences)

d2v_model.train(meta_sentences, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

d2v_model.random.seed(0)
# Saving the model trainee with the orginal dataset

from gensim.test.utils import get_tmpfile

fname = get_tmpfile("my_doc2vec_model")

d2v_model.save(fname)

model = Doc2Vec.load(fname)
# saving the model in a pickle file, because it takes at list half an hour to run...

pickle.dump(model, open("d2v_meta_saved.pkl", "wb"))
# Applying the model to the target smoking

d2v_smoking = []

for i in range(len(smoking_corpus)):

    model.random.seed(0)

    d2v_smoking.append(model.infer_vector(smoking_corpus[i], epochs=200))



d2v_smoking[1]
# Applying the model to the target diabetes

d2v_diabetes = []

for i in range(len(diabetes_corpus)):

    model.random.seed(0)

    d2v_diabetes.append(model.infer_vector(diabetes_corpus[i], epochs=200))



d2v_diabetes[1]
# Applying the model to the target hypertension

d2v_hypertension = []

for i in range(len(hypertension_corpus)):

    model.random.seed(0)

    d2v_hypertension.append(model.infer_vector(hypertension_corpus[i], epochs=200))



d2v_hypertension[1]
# Taking target 1 as an example in the filtered dataset

check = stemmed_meta.title.str.contains(stemmed_smoking.title[1])

print(stemmed_smoking.title[1])

print('check:')

print(stemmed_meta[check])
stemmed_meta[check]
#Cheking the embedding for the model equivalent to the smoking 1 

model[16819]
d2v_smoking[1]
# Checking that the similarity identifies the document on the original dataset

similars = model.docvecs.most_similar(positive=[d2v_smoking[1]], topn=3)

print(similars)
# confirming that the stemmed abstracts are identical #####CHECK HERE#####

print(stemmed_meta['title'][16819])

print(meta_list[16819])

print(smoking_list[1])
# Making a function for getting the top similar to add to the target table



def get_similar_docs(target_df, meta_df, d2v_model, d2v_target):

    """

      This function takes:

      [1] a target table dataframe

      [2] the metadata table dataframe

      [3] doc2vec model based on the metadata abstracts

      [4] doc2vec model of the target table obtained with the metadata doc2vec model



      Both the target and the metadata tables should contain columns: title, abstract and pdf_json_files.



      For this function to run successfully, 

      the following packages need to be installed:

       from gensim.models.doc2vec import Doc2Vec

       import pandas as pd



      At the end it prints the value count of the final dataframes that contains the following columns:

      ('index', 'original_db', 'similarity_percentage', 'title', 'abstract', 'pdf_json_files');



      It mades 3 dataframes:

      * not_target: it contains all the new docs found

      * similar_to_target_df: it contains the original results from the similarity function (target articles + 1st, 2nd and 3rd most similar)

      * new_docs_target_df: it contains the target articles + similar docs that are not in the target table

      

      At the end it returns the new_docs_target_df and the not_target.

      

    """

    # Run the similarity test assuming all titles are in the filtered dataset:

    similar_to_target = []

    for i in range(len(target_df.title)):

        sim_test = d2v_model.docvecs.most_similar(positive=[d2v_target[i]], topn=3)

        #this way the list could be used to create a dataframe

        similar_to_target.append([target_df['index'][i], 'target', 1, 

                                  target_df.title[i], 

                                  target_df.abstract[i], 

                                  target_df.pdf_json_files[i]])

        

        similar_to_target.append([meta_df['index'][sim_test[0][0]], 'most similar', sim_test[0][1], 

                                  meta_df.title[sim_test[0][0]], 

                                  meta_df.abstract[sim_test[0][0]], 

                                  meta_df.pdf_json_files[sim_test[0][0]]])



        #checking if the second and third most similar docs are in target table, if not then append them:

        if meta_df.title[sim_test[1][0]] not in list(target_df.title):

            similar_to_target.append([meta_df['index'][sim_test[1][0]], 'second most similar', sim_test[1][1], 

                                      meta_df.title[sim_test[1][0]], meta_df.abstract[sim_test[1][0]], meta_df.pdf_json_files[sim_test[1][0]]])

        

        elif meta_df.title[sim_test[2][0]] not in list(target_df.title):

            similar_to_target.append([meta_df['index'][sim_test[2][0]], 'third most similar', sim_test[2][1], 

                                      meta_df.title[sim_test[2][0]], meta_df.abstract[sim_test[2][0]], meta_df.pdf_json_files[sim_test[2][0]]])



# creating a dataframe with the top 3 most similar docs of the target ones!

    df_colum = ['original_index', 'original_db', 'similarity_percentage', 'title', 'abstract', 'pdf_json_files']

    similar_to_target_df = pd.DataFrame(similar_to_target, index=range(len(similar_to_target)), columns=df_colum)

    # removing the duplicates

    new_docs_target_df = similar_to_target_df.drop_duplicates(subset='title', keep="first", inplace=False)

    new_docs_target_df.reset_index(drop=True, inplace=True)

    # filtering the target docs, and staying only with the new docs

    not_target = new_docs_target_df[new_docs_target_df['original_db'] !='target']

    not_target.reset_index(drop=True, inplace=True)

    

    print('From the orginal similarity test, we get a total of ' + str(len(similar_to_target_df)) +' articles, counting the target ones and their most similars from metadata.')

    print('After filtering the duplicates from that dataframe, we get a total of ' + str(len(new_docs_target_df)) +' articles.')

    print('Finally, after filtering the target ones, we end with a total of ' + str(len(not_target)) +' new possible articles for the target table.')

    

    #return not_target, similar_to_target_df, new_docs_target_df

    return new_docs_target_df, not_target
def get_relevant_docs(dataframe, target):

    """

    This function will get the relevant docs from a dataframe depending on a target subject.

    

    The dataframe needs to have a column 'pdf_json_files' on it, containing json files names.

    The target is in str format.

    

    At the end it will print the number of relevant docs and 

    return a list of lists for each relevant article with its:

        [0] = original index,

        [1] = original body text 

        [2] = if target or not   

    

    """

    # parsing through all the docs in the target table

    related_docs = []

    

    for i in range(len(dataframe)):

        ori_ind = dataframe.original_index[i]

        ori_tab = dataframe.original_db[i]

        try:

            # open json file

            with open(path + dataframe.pdf_json_files[i], 'r') as myfile:

                data=myfile.read()

            # parse file

            obj = json.loads(data)

            body = obj['body_text']

            # having a list of parts of the text for better parsing

            just_text = [body[d]['text'] for d in range(len(body))]

            clean_body = [text.lower() for text in just_text]

            clean_body = [strip_numeric(text) for text in clean_body] # Remove numbers

            clean_body = [strip_punctuation(text) for text in clean_body] # Remove punctuation

            clean_body = [strip_multiple_whitespaces(text) for text in clean_body] # Remove multiple spaces

            clean_body = [remove_stopwords(text) for text in clean_body] #removing the stopwords

            clean_body = [strip_short(text) for text in clean_body]

            stem_body = [stem_text(text) for text in clean_body]

            relevant_parts = []

            # check if the doc is related with the target

            for t in range(len(stem_body)):

                if target in stem_body[t]:

                    # save the index of relevant parts

                    relevant_parts.append(t)

            # save the docs in a list that has: target_index, clean_json_body, original_json_body

            if len(relevant_parts) != 0:

                # convert json body_text into a text to have the original text 

                original_text=''

                for d in range(len(body)):

                    original_text = original_text+body[d]['text']

                related_docs.append([ori_ind, original_text, ori_tab])

                #related_docs.append([ori_ind, original_text, relevant_parts, clean_body])  

        except:

            TypeError

        

    print('You have ' + str(len(related_docs)) + ' relevant docs')

    return related_docs
def build_relevant_docs_df(meta_df, target_df, relevant_docs_list):

    """

    This function needs the metadata dataframe, the target dataframe and a list of relevant documents.

    

    The list of relevant documents must contain one list for each relevant doc, that 

    has 3 values: [0] = original index,

                  [1] = original body text 

                  [2] = if target or not  

                  

    Finally this function returns a dataframe with all the relevant documents body text obtained from the json file

    and its corresponding columns from the metadata table.

    

    """

    relevant_for_target = []

    for i in range(len(relevant_docs_list)): 

    #this way the list could be used to create a dataframe

        index_rev = relevant_docs_list[i][0]

        #print('from relevant list')

        #print(index_rev)

        

        if relevant_docs_list[i][2] != 'target':

            df_index = list(meta_df[meta_df['index']== index_rev].index)[0]

            relevant_for_target.append([index_rev, 

                                        meta_df.publish_time[df_index],  

                                        meta_df.title[df_index], 

                                        meta_df.abstract[df_index], 

                                        meta_df.cord_uid[df_index], 

                                        meta_df.doi[df_index],

                                        meta_df.journal[df_index],

                                        meta_df.url[df_index],

                                        meta_df.pdf_json_files[df_index],

                                        relevant_docs_list[i][1],

                                        relevant_docs_list[i][2]

                                       ])

        elif relevant_docs_list[i][2] == 'target':

            df_index = list(target_df[target_df['index']== index_rev].index)[0]

            #print('in target table')

            #print(df_index)

            

            relevant_for_target.append([index_rev, 

                                        target_df.Date[df_index],  

                                        target_df.Study[df_index], 

                                        target_df.abstract[df_index], 

                                        target_df.cord_uid[df_index], 

                                        target_df.doi[df_index],

                                        target_df.journal[df_index],

                                        target_df.url[df_index],

                                        target_df.pdf_json_files[df_index],

                                        relevant_docs_list[i][1],

                                        relevant_docs_list[i][2]

                                       ])

        

    # creating a dataframe with the relevant docs including all needed columns

    df_colum = ['original_index', 'publish_time', 'title', 'abstract', 'cord_uid', 'doi',

                    'journal', 'url', 'pdf_json_files', 'body_text', 'target_or_not']

    relevant_df = pd.DataFrame(relevant_for_target, index=range(len(relevant_for_target)), columns=df_colum)

    return relevant_df
# For smoking target

smoking_new_docs, smoking_not_target = get_similar_docs(stemmed_smoking, stemmed_meta, model, d2v_smoking)

smoking_new_docs.head(3)
# For diabetes

diabetes_new_docs, diabetes_not_target = get_similar_docs(stemmed_diabetes, stemmed_meta, model, d2v_diabetes)

diabetes_new_docs.head()

# For hypertension

hyper_new_docs, hyper_not_target = get_similar_docs(stemmed_hypertension, stemmed_meta, model, d2v_hypertension)

hyper_new_docs.head(3)
### Modifying function so that it works with target table dataframe

def get_relevant_docs_mod(dataframe, target):

    """

    This function will get the relevant docs from a dataframe depending on a target subject.

    

    The dataframe needs to have a column 'pdf_json_files' on it, containing json files names.

    The target is in str format.

    

    At the end it will print the number of relevant docs and 

    return a list of lists for each relevant article with its:

        [0] = parts of the body where the target appears

        [1] = clean body   

    

    IT IS MODIFIED SO IT WORKS WITH TARGET TABLE!!

    """

    # parsing through all the docs in the target table

    related_docs = []

    

    for i in range(len(dataframe)):

        try:

            # open json file

            with open(path + dataframe.pdf_json_files[i], 'r') as myfile:

                data=myfile.read()

            # parse file

            obj = json.loads(data)

            body = obj['body_text']

            # having a list of parts of the text for better parsing

            just_text = [body[d]['text'] for d in range(len(body))]

            clean_body = [text.lower() for text in just_text]

            clean_body = [strip_numeric(text) for text in clean_body] # Remove numbers

            clean_body = [strip_punctuation(text) for text in clean_body] # Remove punctuation

            clean_body = [strip_multiple_whitespaces(text) for text in clean_body] # Remove multiple spaces

            clean_body = [remove_stopwords(text) for text in clean_body] #removing the stopwords

            clean_body = [strip_short(text) for text in clean_body]

            stem_body = [stem_text(text) for text in clean_body]

            relevant_parts = []

            # check if the doc is related with the target

            for t in range(len(stem_body)):

                if target in stem_body[t]:

                    # save the index of relevant parts

                    relevant_parts.append(t)

            # save the docs in a list 

            if len(relevant_parts) != 0:

                related_docs.append([relevant_parts, clean_body])   

        except:

            TypeError

            #print('error')

            

    print('You have ' + str(len(related_docs)) + ' relevant docs')

    return related_docs
stem_text('smoking')
# Checking how many json files we have available

print('pdf json files we have available in original target table')

print(stemmed_smoking.pdf_json_files.notna().sum())

print('pdf json files we have available in new target table')

print(smoking_new_docs.pdf_json_files.notna().sum())
# For smoking

print('Original relevant target articles: ')

smoking_target_check = get_relevant_docs_mod(stemmed_smoking, 'smoke')

print('New articles added: ')

smoking_new_relevant = get_relevant_docs(smoking_not_target, 'smoke')

print('Total relevant target articles: ')

smoking_relevant_docs = get_relevant_docs(smoking_new_docs, 'smoke')
stem_text('diabetes')
# Checking how many json files we have available

print('pdf json files we have available in original target table')

print(stemmed_diabetes.pdf_json_files.notna().sum())

print('pdf json files we have available in new target table')

print(diabetes_new_docs.pdf_json_files.notna().sum())
# For diabetes

print('Original relevant target articles: ')

diabetes_target_check = get_relevant_docs_mod(stemmed_diabetes, 'diabet')

print('New articles added: ')

diabetes_new_relevant = get_relevant_docs(diabetes_not_target, 'diabet')

print('Total relevant target articles: ')

diabetes_relevant_docs = get_relevant_docs(diabetes_new_docs, 'diabet')
stem_text('hypertension')
# Checking how many json files we have available

print('pdf json files we have available in original target table')

print(stemmed_hypertension.pdf_json_files.notna().sum())

print('pdf json files we have available in new target table')

print(hyper_new_docs.pdf_json_files.notna().sum())
# For hypertension

print('Original relevant target articles: ')

hyper_target_check = get_relevant_docs_mod(stemmed_hypertension, 'hypertens')

print('New articles added: ')

hyper_new_relevant = get_relevant_docs(hyper_not_target, 'hypertens')

print('Total relevant target articles: ')

hyper_relevant_docs = get_relevant_docs(hyper_new_docs, 'hypertens')
# For smoking

smoking_relevant_df = build_relevant_docs_df(stemmed_meta, stemmed_smoking, smoking_relevant_docs)

smoking_relevant_df.head()

smoking_relevant_df.target_or_not.value_counts()
#pickle.dump(smoking_relevant_df, open("smoking_new_target.pkl", "wb"))
# For diabetes

diabetes_relevant_df = build_relevant_docs_df(stemmed_meta, stemmed_diabetes,diabetes_relevant_docs)

diabetes_relevant_df.head()

diabetes_relevant_df.target_or_not.value_counts()
#pickle.dump(diabetes_relevant_df, open("diabetes_new_target.pkl", "wb"))
# For hypertension

hyper_relevant_df = build_relevant_docs_df(stemmed_meta, stemmed_hypertension,hyper_relevant_docs)

hyper_relevant_df
hyper_relevant_df.target_or_not.value_counts()
#pickle.dump(hyper_relevant_df, open("hypertension_new_target.pkl", "wb"))
#smoke_df = pickle.load(open('smoking_new_target.pkl', 'rb'))

smoke_df = smoking_relevant_df
#diabetes_df = pickle.load(open('diabetes_new_target.pkl', 'rb'))

diabetes_df = diabetes_relevant_df
#hypertension_df = pickle.load(open('hypertension_new_target.pkl', 'rb'))

hypertension_df = hyper_relevant_df
smoke_df['sent']=''     # this column will store tokenized version of the body text

smoke_df['summary']=''  # this column will store the summaries
diabetes_df['sent']=''     # this column will store tokenized version of the body text

diabetes_df['summary']=''  # this column will store the summaries
hypertension_df['sent']=''     # this column will store tokenized version of the body text

hypertension_df['summary']=''  # this column will store the summaries
from nltk.tokenize import sent_tokenize # PASS THIS T THE BEGINNING

for i in range(len(smoke_df.body_text)):

    smoke_df.sent[i]=sent_tokenize(smoke_df.body_text[i])
for i in range(len(diabetes_df.body_text)):

    diabetes_df.sent[i]=sent_tokenize(diabetes_df.body_text[i])
for i in range(len(hypertension_df.body_text)):

    hypertension_df.sent[i]=sent_tokenize(hypertension_df.body_text[i])
def countOccurences(sentence, word):     

    # split the string by spaces in a 

    a = sentence.split()  

    # search for pattern in a 

    count = 0

    for i in range(0, len(a)):           

        # if match found increase count  

        if (word == a[i]): 

            count = count + 1             

    return count 
# identify different stemmed version of smok*

print(stem_text("smoker"))

print(stem_text("smoking"))

print(stem_text("smokers"))

print(stem_text("smoke"))
print(stem_text("diabetes"))
print(stem_text("hypertension"))

print(stem_text("hypertensive"))
for j in range(len(smoke_df.body_text)):

    # d is an internal dataframe for preprocessing on sentence level of the documents and count occurence of relveant words

    d=pd.DataFrame(index=np.arange(len(smoke_df.sent[j])))

    d['content']= 'none'

    d['original']='none'

    d['index']=d.index

    d['count']= 0

    for k in range(len(smoke_df.sent[j])): #iterating through sentences of a document

        d.content[k]=smoke_df.sent[j][k]

        d.original[k]=smoke_df.sent[j][k]

        d.content[k] =  strip_numeric(d.content[k]) #Remove digits

        d.content[k] =  strip_punctuation(d.content[k])  #Remove punctuation

        d.content[k] =  strip_multiple_whitespaces(d.content[k]) #Remove multiple whitespaces   

        d.content[k] = d.content[k].lower() # lower characters

        d.content[k] = remove_stopwords(d.content[k]) #remove stopwords

        d.content[k] = strip_short(d.content[k]) # remove short words

        d.content[k] = stem_text(d.content[k]) #stem the words 

        sentence = d.content[k] # 

        word =stem_text("smoke") # get first relevant word

        word2 =stem_text("smoker") # get second relevant word

        d['count'][k]=float((countOccurences(sentence, word))) +float((countOccurences(sentence, word2))) #storing the amount of relevant words

    x=d.loc[d['count'].idxmax()] # getting the dataframe d with the maximal amount of relevant words   

    smoke_df.summary[j]=x['original'] # storing the sentence with the maximal amount of relevant words in the summary
for j in range(len(diabetes_df.body_text)):

    # d is an internal dataframe for preprocessing on sentence level of the documents and count occurence of relveant words

    d=pd.DataFrame(index=np.arange(len(diabetes_df.sent[j])))

    d['content']= 'none'

    d['original']='none'

    d['index']=d.index

    d['count']= 0

    for k in range(len(diabetes_df.sent[j])): #iterating through sentences of a document

        d.content[k]= diabetes_df.sent[j][k]

        d.original[k]= diabetes_df.sent[j][k]

        d.content[k] =  strip_numeric(d.content[k]) #Remove digits

        d.content[k] =  strip_punctuation(d.content[k])  #Remove punctuation

        d.content[k] =  strip_multiple_whitespaces(d.content[k]) #Remove multiple whitespaces   

        d.content[k] = d.content[k].lower() # lower characters

        d.content[k] = remove_stopwords(d.content[k]) #remove stopwords

        d.content[k] = strip_short(d.content[k]) # remove short words

        d.content[k] = stem_text(d.content[k]) #stem the words 

        sentence = d.content[k] # 

        word =stem_text("diabetes") # get first relevant word     

        d['count'][k]=float((countOccurences(sentence, word)))  #storing the amount of relevant words

    x=d.loc[d['count'].idxmax()] # getting the dataframe d with the maximal amount of relevant words   

    diabetes_df.summary[j]=x['original'] # storing the sentence with the maximal amount of relevant words in the summary

    
for j in range(len(hypertension_df.body_text)):

    # d is an internal dataframe for preprocessing on sentence level of the documents and count occurence of relveant words

    d=pd.DataFrame(index=np.arange(len(hypertension_df.sent[j])))

    d['content']= 'none'

    d['original']='none'

    d['index']=d.index

    d['count']= 0

    for k in range(len(hypertension_df.sent[j])): #iterating through sentences of a document

        d.content[k]= hypertension_df.sent[j][k]

        d.original[k]= hypertension_df.sent[j][k]

        d.content[k] =  strip_numeric(d.content[k]) #Remove digits

        d.content[k] =  strip_punctuation(d.content[k])  #Remove punctuation

        d.content[k] =  strip_multiple_whitespaces(d.content[k]) #Remove multiple whitespaces   

        d.content[k] = d.content[k].lower() # lower characters

        d.content[k] = remove_stopwords(d.content[k]) #remove stopwords

        d.content[k] = strip_short(d.content[k]) # remove short words

        d.content[k] = stem_text(d.content[k]) #stem the words 

        sentence = d.content[k] # 

        word =stem_text("hypertension") # get first relevant word     

        d['count'][k]=float((countOccurences(sentence, word)))  #storing the amount of relevant words

    x=d.loc[d['count'].idxmax()] # getting the dataframe d with the maximal amount of relevant words   

    hypertension_df.summary[j]=x['original'] # storing the sentence with the maximal amount of relevant words in the summary

smoke_df = smoke_df.drop(columns=["sent","body_text"]) 

# body_text and sent are not necessary anymore
diabetes_df = diabetes_df.drop(columns=["sent","body_text"]) 

# body_text and sent are not necessary anymore
hypertension_df = hypertension_df.drop(columns=["sent","body_text"]) 

# body_text and sent are not necessary anymore
smoke_df.head()
diabetes_df.head()
hypertension_df.head()
for i in range(len(smoke_df)):

    print('')

    print("Title:",smoke_df.title[i],"," ,smoke_df.publish_time[i], "(",smoke_df.target_or_not[i], ")")

    print('')

    print("Summary:",smoke_df.summary[i])

    print('')
for i in range(len(diabetes_df)):

    print('')

    print("Title:",diabetes_df.title[i],"," ,diabetes_df.publish_time[i], "(",diabetes_df.target_or_not[i], ")")

    print('')

    print("Summary:",diabetes_df.summary[i])

    print('')
for i in range(len(hypertension_df)):

    print('')

    print("Title:",hypertension_df.title[i],"," ,hypertension_df.publish_time[i], "(",hypertension_df.target_or_not[i], ")")

    print('')

    print("Summary:",hypertension_df.summary[i])

    print('')