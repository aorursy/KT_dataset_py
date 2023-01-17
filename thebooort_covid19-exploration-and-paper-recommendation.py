import numpy as np 

import pandas as pd 

import os

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis.sklearn

import json

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image

import re

import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")
os.listdir('/kaggle/input/CORD-19-research-challenge/')




with open('/kaggle/input/CORD-19-research-challenge/metadata.readme', 'r') as f:

    data = f.read()

    print(data)





dirs = ['/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/',

        '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/',

        '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/',

        '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/']



filenames=[]

docs =[]

for d in dirs:

    for file in os.listdir(d):

        filename = d +file

        j = json.load(open(filename, 'rb'))

        

        paper_id =j['paper_id']

        #date =j['date']

        title = j['metadata']['title']

        authors = j['metadata']['authors']

        list_authors =[]

        for author in authors:

            if(len(author['middle'])==0):

                middle =""

            else :

                middle = author['middle'][0]

            _authors =author['first']+ " "+ middle +" "+ author['last']

            list_authors.append(_authors)

            

        try :

            abstract =  j['abstract'][0]['text']

        except :

            abstract =" "

        

        full_text =""

        for text in  j['body_text']:

            full_text += text['text']

        

        docs.append([paper_id,title,list_authors,abstract,full_text])



df = pd.DataFrame(docs,columns=['paper_id','title','list_authors','abstract','full_text'])

df.to_csv('/kaggle/working/data.csv')

df.head()
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(x.strip().split()))

df['body_word_count'] = df['full_text'].apply(lambda x: len(x.strip().split()))

df.head()
df.shape



df.isnull().sum()
#df.dropna(inplace=True,axis=0)
for key in ['abstract','title','full_text']:

    total_words = df[key].values

    wordcloud = WordCloud(width=1800, height=1200).generate(str(total_words))

    plt.figure( figsize=(30,10) )

    plt.title ('Wordcloud' + key)

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()
df.size



df[['abstract_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))

plt.show()



df[['body_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))

plt.show()
print(df['abstract_word_count'].mean(),df['abstract_word_count'].std())
print(df['body_word_count'].mean(),df['body_word_count'].std())
features  = 5000

# TODO: probar con TFIDF

tf_vectorizer = CountVectorizer(max_features=features, stop_words='english', min_df=10)

X_tf = tf_vectorizer.fit_transform(df['abstract'])

tf_feat_name = tf_vectorizer.get_feature_names()
topics = 7

lda_model = LatentDirichletAllocation(learning_method='online',random_state=23,n_components=topics)

lda_output =lda_model.fit_transform(X_tf)
# preparing for plotting pyLDAvis

%matplotlib inline

pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda_model, X_tf, tf_vectorizer)

# if you want to save it 

# P=pyLDAvis.sklearn.prepare(lda_model, X_tf, tf_vectorizer)

# pyLDAvis.save_html(p, 'lda.html')
# func from Anish Pandey

def visualizing_topic_cluster(model,stop_len,feat_name):

    topic={}

    for index,topix in enumerate(model.components_):

        topic[index]= [feat_name[i] for i in topix.argsort()[:-stop_len-1:-1]]

    

    return topic



topic_lda =visualizing_topic_cluster(lda_model,10,tf_feat_name)

# printing 

len([print('Topic '+str(key),topic_lda[key]) for key in topic_lda])
# dirty as fuck, removing words that appears in same topic

import copy

topic_lda_2 = topic_lda

for key in topic_lda_2:

    for element in topic_lda_2[key]:

        if element in ['cell','cells','viral','virus','respiratory','study','infection','acute']:

            topic_lda_2[key].remove(str(element))

[print('Topic '+str(key),topic_lda_2[key]) for key in topic_lda_2]
# lets see if our topics are correlated, first mixing topics with our df

columns=['Topic'+ i for i in list(map(str,list(topic_lda.keys())))]

ldadf =pd.DataFrame(lda_output,columns=columns).apply(lambda x : np.round(x,3))

ldadf['Major_topic'] =lda_df[columns].idxmax(axis=1).apply(lambda x: int(x[-1]))

ldadf['keyword'] = lda_df['Major_topic'].apply(lambda x: topic_lda[x])

ldadf.head()



# plotting results

import seaborn as sns

plt.figure(figsize=(10,10))

sns.heatmap(abs(lda_df[columns].corr()),annot=True,fmt ='0.2f',cmap="YlGnBu")

plt.title(" Correlation Plot")