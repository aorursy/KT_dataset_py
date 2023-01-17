import numpy as np 

import pandas as pd 

pd.set_option('max_colwidth',400)

import matplotlib.pyplot as plt

import json

import pickle

#test get business data

biz_f = open('../input/yelp_academic_dataset_business.json')

biz_df = pd.DataFrame([json.loads(x) for x in biz_f.readlines()])

biz_f.close()
biz_df=biz_df.dropna()
biz_df.head(3)
restaurant = biz_df[biz_df.apply(lambda x: 'Restaurants' in x['categories'], axis=1)]
restaurant.shape
# Load Yelp reviews data

review_file = open('../input/yelp_academic_dataset_review.json')

review_df = pd.DataFrame([json.loads(next(review_file)) for x in range(biz_df.shape[0])])

review_file.close()
review_df.head()
#only use 2 years data

review_df=review_df[review_df.date>='2017-01-01']
review_df.shape
#join dataframe

restaurant_reviews = restaurant.merge(review_df, on='business_id', how='inner')
r_reviews=restaurant_reviews[['stars_y','categories','text']]
r_reviews.shape
r_reviews.head()
r_reviews.groupby('categories').size().reset_index(name='count').sort_values('count', ascending=False)
import re

import string
def clean_text_round1(text):

    text=text.lower()

    text=re.sub('\[.*?\]','',text)

    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)

    text=re.sub('\w*\d\w*','',text)

    return text

round1=lambda x: clean_text_round1(x)
r_reviews['text']=r_reviews.text.apply(round1)
def clean_text_round2(text):

    text=re.sub('[^A-Za-z0-9]+', ' ',text)

    return text



round2= lambda x: clean_text_round2(x)
r_reviews['text']=r_reviews.text.apply(round2)
r_reviews.text.head()
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

import nltk

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem import *
text1 = " ".join(log for log in r_reviews.text)
stopwords=gensim.parsing.preprocessing.STOPWORDS
#exploring dataset

from wordcloud import WordCloud

#use default stopwords

wc = WordCloud(stopwords=stopwords, background_color="white", colormap="Dark2",

               max_font_size=150, random_state=42)
wordcloud =wc.generate(text1)

plt.rcParams['figure.figsize'] = [16, 7]

plt.imshow(wordcloud, interpolation='bilinear')    

plt.subplot(1,1,1)

plt.axis("off")

plt.title('word cloud')
#further clean with stop words

r_reviews.text = r_reviews.text.apply(lambda x:' '.join(x for x in x.split() if not x in stopwords))
#approach 1: using Doc2vec+k-means clustering

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
LabeledSentence1 = gensim.models.doc2vec.TaggedDocument

all_content_train = []

j=0

for em in r_reviews['text'].values:

    all_content_train.append(LabeledSentence1(em,[j]))

    j+=1

print('Number of texts processed: ', j)
model = Doc2Vec(vector_size=100, dbow_words= 1, dm=0, epochs=1,  window=5, seed=42, min_count=5, workers=40,alpha=0.025, min_alpha=0.025)
model.build_vocab(all_content_train)
count = len(r_reviews)

for epoch in range(10):

    if epoch%5 == 0:

        print("epoch "+str(epoch))

    model.train(all_content_train, total_examples=count, epochs=1)

    model.save('doc2vec.model')

    if epoch%5 == 0:

        model.alpha -= 0.002  # decrease the learning rate

        model.min_alpha = model.alpha 
fname = "doc2vec.model"

model = Doc2Vec.load(fname)
from sklearn.cluster import KMeans
#elbow evealuate

test1=np.arange(start=2,stop=30,step=2)

clustering_1=pd.DataFrame()

for t in test1:

    kmeans=KMeans(n_clusters=t,random_state=42)

    kmeans.fit(model.docvecs.vectors_docs)

    df_temp=pd.DataFrame({'clusters':[t],'sse':[kmeans.inertia_]})

    clustering_1=clustering_1.append(df_temp)
#plot results

plt.figure(figsize=(6,5))

plt.scatter(clustering_1['clusters'],clustering_1['sse'])

plt.title('clustering group')
#test with 6 clusters

kmeans_model=KMeans(n_clusters=6, init='k-means++',max_iter=100)

kmeans_model.fit(model.docvecs.vectors_docs)

clusters = kmeans_model.labels_.tolist()
r_reviews['doc2vec_label']=clusters
plt.figure(figsize=(7,5))

plt.bar(range(0,6), [r_reviews['doc2vec_label'].value_counts()[i] for i in range(0, 6)], align='center', alpha=0.5)

plt.ylabel('Number of texts')

plt.xlabel('Cluster Number')

plt.title('Number of texts in each cluster')

plt.show()
#getting topic words

from collections import Counter

print('top terms per cluster:')

print()

order_centroids=kmeans_model.cluster_centers_.argsort()[:,::-1]

for i in range(0,6):

    titles=r_reviews[r_reviews['doc2vec_label']==i]

    words=[x for x in titles['text']]

    count = Counter(' '.join(words))

    count = Counter(' '.join(words).split(' '))

    s = ''

    print('Cluster '+str(i)+' words:')



    for i in  count.most_common()[:20]:

        s += i[0]+" "

    print(s+'\n')
#visualize with PCA

from sklearn.decomposition import PCA

pca=PCA(n_components=2).fit(model.docvecs.vectors_docs)
datapoint=pca.transform(model.docvecs.vectors_docs)
label=kmeans_model.labels_
centroids=kmeans_model.cluster_centers_

centroidpoint=pca.transform(centroids)
import seaborn as sns

plt.figure(figsize=(7,7))

sns.scatterplot(datapoint[:,0],datapoint[:,1],hue=label)

sns.scatterplot(centroidpoint[:, 0], centroidpoint[:, 1], marker='^',s=150)
texts=r_reviews['text'].tolist()
stemmer = SnowballStemmer("english")

def tokenize_only(text):

    filtered_tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    return filtered_tokens
def tokenize_and_stem(text):

    filtered_tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems
totalvocab_stemmed = []

totalvocab_tokenized = []

for i in texts:

    allwords_stemmed = tokenize_and_stem(i)

    totalvocab_stemmed.extend(allwords_stemmed)

    

    allwords_tokenized = tokenize_only(i)

    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,

                                 min_df=10, stop_words='english',

                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))



%time tfidf_matrix = tfidf_vectorizer.fit_transform(texts)



print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
num_clusters = 6

km = KMeans(n_jobs=-1, n_clusters=num_clusters, random_state = 42)

%time km.fit(tfidf_matrix)

TF_cluster = km.labels_.tolist()
r_reviews['TF_cluster']=TF_cluster
print("Top terms per cluster:")

print()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(6):

    print('Cluster '+str(i)+' words:')

    s = ""

    for ind in order_centroids[i, :40]:

        s+=str(vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])+' '

    print(s)

    print( )
r_reviews_c5=r_reviews.loc[r_reviews.TF_cluster==5].groupby('categories').size().reset_index(name='count').sort_values('count', ascending=False)

r_reviews_c5
plt.bar(range(0,6),[r_reviews['TF_cluster'].value_counts()[i] for i in range(0,6)],align='center',alpha=0.5)

plt.ylabel('Num of texts')

plt.xlabel('Cluster num')

plt.title('number of text in each cluster')

plt.show()                     
#Approach 3 bag of words--LDA
tokenized_text=[tokenize_and_stem(text)for text in texts]

texts=[[word for word in text if word not in stopwords] for text in tokenized_text]
from gensim import corpora,models,similarities

dictionary=corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5,no_above=0.8)

corpus=[dictionary.doc2bow(text) for text in texts]
lda=models.LdaMulticore(corpus,num_topics=6,workers=40,id2word=dictionary,chunksize=10000,

                       passes=100)
for idx,topic in lda.print_topics(-1):

    print('Topic: {} Word: {}'.format(idx,topic))
#get result out

sent_topics_df = pd.DataFrame()

for i,row_list in enumerate(lda[corpus]):

    row=row_list[0] if lda.per_word_topics else row_list

    row=sorted(row,key=lambda x:(x[1]),reverse=True)

    for j,(topic_num,prop_topic) in enumerate(row):

        if j==0:

            wp=lda.show_topic(topic_num)

            topic_keywords=",".join([word for word,prop in wp])

            sent_topics_df=sent_topics_df.append(pd.Series([int(topic_num),round(prop_topic,4),topic_keywords]),ignore_index=True)

        else: 

            break

sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
plt.figure(figsize=(7,7))

plt.bar(range(0,6),[sent_topics_df.Dominant_Topic.value_counts()[i] for i in range(0,6)],align='center',alpha=0.5)

plt.ylabel('Num of texts')

plt.xlabel('Cluster num')

plt.title('number of text in dominent topics')

plt.show()  
import pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
viz=pyLDAvis.gensim.prepare(lda,corpus,dictionary)

viz
pyLDAvis.save_html(viz, 'lda.html')
#create labels as per stars, <=3 label as 0, >3 label as 1

labels=r_reviews['stars_y'].map(lambda x : 1 if int(x) > 3 else 0)
#test the difference

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

vocabulary_size = 20000

tokenizer = Tokenizer(num_words= vocabulary_size)

tokenizer.fit_on_texts(r_reviews['text'])



sequences = tokenizer.texts_to_sequences(r_reviews['text'])

data = pad_sequences(sequences, maxlen=50)
print(data.shape)
from keras.models import Sequential

from keras.layers import Bidirectional, Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding
model_lstm = Sequential()

model_lstm.add(Embedding(20000, 100, input_length=50))

model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()
model_lstm.fit(data, np.array(labels), validation_split=0.4, epochs=3)
## Plot

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

import matplotlib as plt
word_list = []

for word, i in tokenizer.word_index.items():

    word_list.append(word)



def plot_words(data, start, stop, step):

    trace = go.Scatter(

        x = data[start:stop:step,0], 

        y = data[start:stop:step, 1],

        mode = 'markers',

        text= word_list[start:stop:step]

    )

    layout = dict(title= 't-SNE 1 vs t-SNE 2',

                  yaxis = dict(title='t-SNE 2'),

                  xaxis = dict(title='t-SNE 1'),

                  hovermode= 'closest')

    fig = dict(data = [trace], layout= layout)

    py.iplot(fig)
from sklearn.manifold import TSNE





lstm_embds = model_lstm.layers[0].get_weights()[0]

lstm_tsne_embds = TSNE(n_components=2).fit_transform(lstm_embds)
plot_words(lstm_tsne_embds, 0, 20000, 1)
def create_conv_model():

    model_conv = Sequential()

    model_conv.add(Embedding(vocabulary_size, 100, input_length=50))

    model_conv.add(Dropout(0.2))

    model_conv.add(Conv1D(64, 5, activation='relu'))

    model_conv.add(MaxPooling1D(pool_size=4))

    model_conv.add(LSTM(100))

    model_conv.add(Dense(1, activation='sigmoid'))

    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_conv
model_conv = create_conv_model()

model_conv.fit(data, np.array(labels), validation_split=0.4, epochs = 3)
conv_embds = model_conv.layers[0].get_weights()[0]

conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)
plot_words(conv_tsne_embds, 0, 2000, 1)
model_bilstm = Sequential()

model_bilstm.add(Embedding(20000, 100, input_length=50))

model_bilstm.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))

model_bilstm.add(Dense(1, activation='sigmoid'))

model_bilstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_bilstm.summary()
model_bilstm.fit(data, np.array(labels), validation_split=0.4, epochs=3)
bilstm_embds = model_bilstm.layers[0].get_weights()[0]

bilstm_tsne_embds = TSNE(n_components=2).fit_transform(bilstm_embds)
plot_words(bilstm_tsne_embds, 0, 2000, 1)