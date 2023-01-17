# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import STOPWORDS, WordCloud

import gensim

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data=pd.read_csv('../input/qa.csv')



# Any results you write to the current directory are saved as output.
##Create function to tokenize and count

def get_count(x):

    return len(nltk.word_tokenize(x))
data['Response_Length']=data['Comey Response'].map(get_count)
data.groupby('Party Affiliation',as_index=False)['Response_Length'].mean().rename(columns={'Response_Length':"AverageWordsUsed"})
g= sns.FacetGrid(data,col="Party Affiliation")

g=g.map(plt.hist,"Response_Length")
## The distribution of responses seems skewed for all party types
data[data['Response_Length']==np.max(data['Response_Length'])][['Senator','Party Affiliation','Full Question']]
data[data['Response_Length']==np.max(data['Response_Length'])]['Full Question'].tolist()
from nltk.corpus import stopwords

from nltk.tokenize import wordpunct_tokenize

stop=set(stopwords.words('english'))

stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

def get_stop_removed(x):

    return [i.lower() for i in wordpunct_tokenize( "".join(x.tolist())) if i.lower() not in stop]
resp_republican=get_stop_removed(data[data['Party Affiliation']=='Republican']['Comey Response'])

resp_democrats=get_stop_removed(data[data['Party Affiliation']=='Democrat']['Comey Response'])

resp_independent=get_stop_removed(data[data['Party Affiliation']=='Independent']['Comey Response'])
wordcloud_rep=WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(" ".join(resp_republican))

wordcloud_dem=WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(" ".join(resp_democrats))

wordcloud_ind=WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(" ".join(resp_independent))
def cloud_plot(wordcloud):

    fig = plt.figure(1, figsize=(20,15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
cloud_plot(wordcloud_rep)
cloud_plot(wordcloud_dem)
cloud_plot(wordcloud_ind)
def get_stop_removed_sent(x):

    return [i.lower() for i in wordpunct_tokenize( "".join(x)) if i.lower() not in stop]

sentences=[]

for w in data['Comey Response']:

    sentences.append(get_stop_removed_sent(w))

    

model=gensim.models.Word2Vec(sentences,min_count=1)
model.most_similar('russia') 
model.most_similar('america')
model.most_similar('fbi')
model.most_similar('president')
corpus=[]

for w in data['Full Question'].tolist():

    corpus.append(w)

tf=TfidfVectorizer(corpus,stop_words='english')

tfidf=tf.fit_transform(data['Full Question'])

len(tf.get_feature_names())
y=data['Party Affiliation'].map({'Republican':0,'Democrat':1,'Independent':2})

lda=LinearDiscriminantAnalysis(n_components=2)

lda.fit(tfidf.todense(),y)

lda_x=lda.transform(tfidf.todense())

plot_data=pd.DataFrame({'Comp1':lda_x[:,0],'Comp2':lda_x[:,1],'Affiliation':data['Party Affiliation']})
sns.lmplot(x='Comp1',y='Comp2',hue='Affiliation',data=plot_data,fit_reg=False)
lda_comp1=pd.Series(lda.coef_.transpose()[:,0],index=tf.get_feature_names())

lda_comp2=pd.Series(lda.coef_.transpose()[:,1],index=tf.get_feature_names())
lda_comp1.sort_values(ascending=False).head(10)
lda_comp1.sort_values(ascending=True).head(10)
lda_comp2.sort_values(ascending=False).head(10)
lda_comp2.sort_values(ascending=True).head(10)