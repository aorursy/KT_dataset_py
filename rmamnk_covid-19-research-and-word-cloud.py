# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt



import os



import spacy



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['year'] = df['publish_time'].dt.year
df['year'].fillna(2020,inplace=True)
df = df.assign(crisis=df['year']>2018)
df['decade'] = df['year'].astype('int')/10
df['decade'] = df['decade'].astype('int')
df['decade'] = df['decade'] * 10
d = pd.read_csv('../input/journal-rank/scimagojr 2018.csv',sep=';')
d = d[['Rank','Title']]
d.columns = ['Rank','journal']
df = df.merge(d,on='journal')
df.drop('journal',axis=1,inplace=True)
df.drop('publish_time',axis=1,inplace=True)
df.drop('source_x',axis=1,inplace=True)
nlp = spacy.load('en')
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms_q1 = ['incubation','Contagious']

patterns = [nlp(text) for text in terms_q1]

matcher.add("incubation", None, *patterns)
terms_q2 = ['asymptomatic']

patterns = [nlp(text) for text in terms_q2]

matcher.add("asymptomatic", None, *patterns)
terms_q3 = ['season']

patterns = [nlp(text) for text in terms_q3]

matcher.add("season", None, *patterns)
terms_q4 = ['physics']

patterns = [nlp(text) for text in terms_q4]

matcher.add("physics", None, *patterns)
terms_q5 = ['Persistence','stability','substrates','sources']

patterns = [nlp(text) for text in terms_q5]

matcher.add("strength", None, *patterns)
terms_q6 = ['material','copper','glass','steel','plastic']

patterns = [nlp(text) for text in terms_q6]

matcher.add("materials", None, *patterns)
terms_q7 = ['history','shedding']

patterns = [nlp(text) for text in terms_q7]

matcher.add("history", None, *patterns)
terms_q8 = ['diagnostics']

patterns = [nlp(text) for text in terms_q8]

matcher.add("diagnostics", None, *patterns)
terms_q9 = ['model']

patterns = [nlp(text) for text in terms_q9]

matcher.add("model", None, *patterns)
terms_q10 = ['phenotypic','adaptation']

patterns = [nlp(text) for text in terms_q10]

matcher.add("survival", None, *patterns)
terms_q11 = ['immune']

patterns = [nlp(text) for text in terms_q11]

matcher.add("immunity", None, *patterns)
terms_q13 = ['enviornment']

patterns = [nlp(text) for text in terms_q13]

matcher.add("enviornment", None, *patterns)
df['title'].fillna('',inplace=True)
df['abstract'].fillna('',inplace=True)
new_df = pd.DataFrame(columns=['title','abstract','decade','crisis','authors','Rank','topic','topic_rank'])
for _,row in df.iterrows():

    rank = row['Rank']

    title = row['title']

    

    abstract = row['abstract']

   

    decade = row['decade']

    crisis = row['crisis']

    authors = row['authors']

    title_doc = nlp(title)

    abstract_doc = nlp(abstract)

    p_matches = matcher(title_doc)

    s_matches = matcher(abstract_doc)

    for m in p_matches:

        match_id ,_,_ = m

        d = pd.DataFrame({'title':title,'abstract':abstract,'decade':decade,'crisis':crisis,'authors':authors,'Rank':rank,'topic':nlp.vocab.strings[match_id],'topic_rank':True},index=[0])

        new_df = pd.concat([d,new_df])

    for m in s_matches:

        match_id ,_,_ = m

        d = pd.DataFrame({'title':title,'abstract':abstract,'decade':decade,'crisis':crisis,'authors':authors,'Rank':rank,'topic':nlp.vocab.strings[match_id],'topic_rank':False},index=[0])

        new_df = pd.concat([d,new_df])
from wordcloud import WordCloud
df = new_df.copy()
df.drop('topic_rank',inplace=True,axis=1)
df.drop_duplicates(inplace=True)
df['Rank'].max()
"""

Python script to generate word cloud image.

Author - Anurag Rana

Read more on - https://www.pythoncircle.com

"""



from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords

 

stop_words=set(stopwords.words('english'))

# image configurations

background_color = "#101010"

height = 720

width = 1080











topics = df['topic'].unique()

for topic in topics:

    data = dict()

    mydf = df[df['topic']==topic]

    for abstract,rank in zip(mydf['abstract'],mydf['Rank']):

        for word in nlp(abstract):

            if word.lemma_ in stop_words:

                continue



            data[word.lemma_] = data.get(word, 0) + (1 - rank/31933 )

    word_cloud = WordCloud(

    background_color=background_color,

    width=width,

    height=height

)

    print(topic)

    wc = word_cloud.generate_from_frequencies(data)

    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')





    plt.savefig('plt_'+topic+'.png')
df.sort_values(by=['Rank','crisis','decade'],inplace=True)
df.to_csv('titles.csv')