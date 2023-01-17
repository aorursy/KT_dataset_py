import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#Dataset files

file1 = '../input/winemag-data-130k-v2.csv'

file2 = '../input/winemag-data_first150k.csv' 





#Read files

f1 = pd.read_csv(file1)

f2 = pd.read_csv(file2)



#To convet dataset into a single dataframe

df1 = pd.DataFrame(f1)

df2 = pd.DataFrame(f2)

frames = [df1, df2]

dfd= pd.concat(frames)



#To re-index de dataframe

##Create a list of number like a new column

dfd['index'] = range(0, len(dfd))

##Set a copy of this column like the index

dfd.set_index(dfd['index'], inplace=True)

##Drop the column that we have previously created

dfd=dfd.drop(columns=['index'])





#To transform variables as 'category type'



dfd['variety']=dfd['variety'].astype('category')

dfd['country']=dfd['country'].astype('category')

dfd['winery']=dfd['winery'].astype('category')



LDF = [df1, df2, dfd]

    

for i in LDF:

    print( i.shape)

    

for i in LDF:

    print( i.columns)

print ('\n \n')



print('DFD info \n')





print(dfd.info())
pd.options.display.max_colwidth = 500





dfd.loc[[132],['description','designation','variety']]
# Dataset slice filtering by one variety ('Cabernet Sauvignon'), and swhowing two variables (variety and description)

#dfd.loc[dfd['variety']=='Cabernet Sauvignon',['variety','points','description']]
dfdv=dfd.loc[:,['description','country']]



#LC = List of countries

LC = dfdv.groupby('country').count()



DLC = pd.DataFrame(LC)

DLC['Country']=DLC.index

DLC['index1'] = range(0, len(DLC))

DLC.set_index(DLC['index1'], inplace=True)

DLC=DLC.drop(columns=['index1'])



DLC.rename(columns={'description': 'QR'}, inplace=True)
print(DLC.head())
price=dfd['price'], 

points=dfd['points']

dfd[dfd['price'] < 4000].plot.scatter(x='price', y='points')


fig, axarr = plt.subplots(2, 2, figsize=(14, 10))

#Gráph 1

dfd['price'].value_counts().sort_index().plot.hist(

    ax=axarr[0][0]

)

## Títle

axarr[0][0].set_title("Wine Price", fontsize=18)



#Graph 2

dfd['points'].value_counts().sort_index().plot.bar(

    ax=axarr[0][1]

)

## Title

axarr[0][1].set_title("Score Distribution", fontsize=18)



#Graph 3

dfd['country'].value_counts().head(20).plot.bar(

   ax=axarr[1][0]

)

## Title

axarr[1][0].set_title("Countries", fontsize=18)





dfd['variety'].value_counts().head(30).plot.bar(

    ax=axarr[1][1]

)

## Title

axarr[1][1].set_title("Vairety Distribution", fontsize=18)
#Libraries needed

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from nltk.tokenize import RegexpTokenizer

from nltk.stem.snowball import SnowballStemmer

import spacy

nlp = spacy.load('en_core_web_sm')
#To identify the duplicated description

DD= dfd[dfd['description'].duplicated(keep = False)].sort_values('description')



print( dfd.shape)

print(DD.shape)



# To drop out the duplicated description to avoid be part of the tokenizen process



dfd = dfd.drop_duplicates('description')
#To filter by those varieties which have at least 1500 description

variety_df = dfd.groupby('variety').filter(lambda x: len(x) > 1500)

varieties=variety_df['variety'].unique()

print('Number of relevant varieties:', varieties.shape)

#To create a bargrpah with the number of description by variety

fig, ax = plt.subplots(figsize = (25, 10))

sns.countplot(x = variety_df['variety'], order = varieties, ax = ax)

plt.xticks(rotation = 90)

plt.show()
#To convert df['description'] into documents that can be used by Spacy

doc = variety_df['description'].values



doc2=variety_df['description']
print(doc.shape)
# Import countvectorizr

from sklearn.feature_extraction.text import  CountVectorizer



#To create a instance of CV in which de words that show up in the 80% of documents are removed and it's needed that the word

#at leasts appear in two documents

cv = CountVectorizer(max_df=0.8, min_df=2, stop_words='english' )
#To apply CV over doc --> Document Text Matrix



dtm = cv.fit_transform(doc)



dtm 
from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components = 10, random_state=42)
LDA.fit(dtm)
# To check the lenth of the vocabulary set
len(cv.get_feature_names())
cv.get_feature_names()[890]
#To Check topics
#To define first topic

first_topic = LDA.components_[0]

#To order the word by index position from leat to great (according relevance)

first_topic.argsort()
# To watch al the topics:



for i, topic in enumerate (LDA.components_):

    print(f"The top 15 words for topic #{i}")

    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])

    print('\n')

    print('\n')
#To define the beloging to a particular topic

Topic_results = LDA.transform(dtm)
print(Topic_results[14].round(2))

print(Topic_results[14].argmax())
VDF= pd.DataFrame(variety_df)
VDF['Topic']=Topic_results.argmax(axis=1)
PN = VDF[VDF['variety']=='Pinot Noir']
len(PN['Topic'].unique())
clusters = VDF.groupby(['Topic', 'variety']).size()

fig2, ax2 = plt.subplots(figsize = (30, 15))

sns.heatmap(clusters.unstack(level = 'variety'), ax = ax2, cmap = 'Reds')