import pandas as pd

#import numpy as np

import matplotlib.pyplot as plt

#import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.corpus import stopwords

# from nltk.stem import WordNetLemmatizer

# from nltk.stem import PorterStemmer

#from sklearn.decomposition import PCA

#from sklearn.decomposition import TruncatedSVD

import nltk

import time

import re

import gensim

#import paramiko

from bs4 import BeautifulSoup as BS

from wordcloud import WordCloud, STOPWORDS 
custom=set()

with open('../input/custom-words/Keywords.txt','r') as file:

    for line in file:

        if len(line.strip())>=3:

            custom.add(line.strip())
def preprocess_df(dataframe,name,html,category):

    dataframe.drop_duplicates(subset=name,keep='first',inplace=True)

    dataframe.dropna(subset=[html,category],inplace=True)

    

    return dataframe

    
def Preprocess(string): 

    #html to description

    soup=BS(string,'html.parser')

    string=soup.text

    

    #punctuation

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    

    #stopwords

    stop_corpus=stopwords.words('english')+list(custom)



    #removing punctuation

    for x in string.lower(): 

        if x in punctuations: 

            string = string.replace(x, "")

            

    string=string.lower()

    

    #removing extra whitespace

    string=' '.join(string.split())

    

    #removing digit

    string=''.join([i for i in string if not i.isdigit()])

    

    cor=[]

    for word in string.split():

        if len(word)>=3:

            if word not in stop_corpus:

                cor.append(word)

                

    string=' '.join(cor)

#     #removing small words

#     string=' '.join(word for word in string.split() if len(word)>=3)

    

#     #removing stopwords

#     string=' '.join([word for word in string.split() if word not in stop_corpus])



    return string
def Visualise(text):

    wordcloud = WordCloud(width=500,height=500,background_color='white',stopwords=STOPWORDS,min_font_size=15)

    wordcloud = wordcloud.generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    
def TaggedDoc(dataframe):

    tagdoc=[]

    for i in range(len(dataframe)):

        X=dataframe.iloc[i]

        y=gensim.models.doc2vec.TaggedDocument(X.lower().split(), [i])

        tagdoc.append(y)

        

    return tagdoc
df=pd.read_csv("../input/qiagen-detail/Qiagen_details.csv",encoding='utf-8')
# df = df.sample(frac=0.4).reset_index(drop=True)
df.head()
df=preprocess_df(df,'product_name','product_html','product_category')
for i in range(len(df['product_html'])):

    df['product_html'].iloc[i]=Preprocess(df['product_html'].iloc[i])
# df.to_csv('product_description.csv',index=False)
df['product_html'].iloc[0]
Visualise(df['product_html'].iloc[0])
df_desc=TaggedDoc(df['product_html'])
len(df)
df['product_category'].value_counts()
cat=list(df['product_category'])
name=list(df['product_name'])
ist=[]

for i in range(len(df_desc)):

    ist.append(df_desc[i].words)
model=gensim.models.doc2vec.Doc2Vec(vector_size=50,epochs=30,dm=0,workers=4)
model.build_vocab(df_desc)
#model.wv.vocab.keys()
len(model.wv.vocab)
%time model.train(df_desc, total_examples=model.corpus_count, epochs=model.epochs)
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 8))  

plt.title("Dendrograms")  

%time dend = shc.dendrogram(shc.linkage(model.docvecs.vectors_docs, method='ward'))
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward') 

#labels=cluster.labels_.tolist()

%time clusters=cluster.fit_predict(model.docvecs.vectors_docs)
df1=pd.DataFrame({

    'cluster':clusters,

    'category':cat,

    'name':name,

    'keywords':ist

})
for i in range(len(df1)):

    df1['keywords'].iloc[i]=','.join(df1['keywords'].iloc[i])
df1['keywords'].iloc[0]
g=df1.groupby(['cluster'])
g.get_group(0)['category'].value_counts()
df1.to_csv('Document_Clustered.csv',index=False)