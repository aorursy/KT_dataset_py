# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/nlp-getting-started/train.csv")
print(data.head())
text_corpus=data['text']
print(text_corpus)
train_data=pd.read_csv("../input/nlp-getting-started/train.csv")
test_data=pd.read_csv("../input/nlp-getting-started/test.csv")
import re

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD,SparsePCA
from sklearn.metrics import classification_report,confusion_matrix
from nltk.tokenize import word_tokenize
from collections import defaultdict
from collections import Counter
stop=set(stopwords.words('english'))
import tensorflow as tf
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from sklearn.manifold import TSNE
from tqdm import tqdm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout,Conv1D,Flatten,Dropout,Activation,MaxPooling1D
from keras.initializers import Constant
from keras.optimizers import Adam

print(data.shape[0])
print(data.shape[1])

print(train_data.shape[0])
print(test_data.shape[0])

print(train_data.columns)
real_tweet=train_data[train_data['target']==1].shape[0]
false_tweet= train_data[train_data['target']==0].shape[0]
print("Real tweets:",real_tweet)
print("Fake tweets:",false_tweet)

def word_freq(data):
    return len(data)

real_tweet_length= train_data[train_data['target']==1]['text'].str.split().map(lambda x: word_freq(x))
false_tweet_length=train_data[train_data['target']==0]['text'].str.split().map(lambda x: word_freq(x))
print(real_tweet_length)
print(false_tweet_length)

def draw_countoftweets(real_tweet,false_tweet):
    plt.rcParams['figure.figsize']=(10,10)
    plt.bar(0,real_tweet,width=0.7,label='Real Tweets',color='Green')
    plt.legend()
    plt.bar(2,false_tweet,width=0.7,label='False Tweets',color='Red')
    plt.legend()
    plt.ylabel('Count of Tweets')
    
    plt.title('Types of Tweets')
    plt.show()

draw_countoftweets(real_tweet,false_tweet)

def draw_words_in_tweet(real_tweet_length,false_tweet_length):
    figs,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    ax1.hist=plt.hist(real_tweet_length,color='Blue')
    ax1.set_title('Real Tweets')
    ax2.hist=plt.hist(false_tweet_length,color='Red')
    ax2.set_title('False Tweets')
    figs.suptitle('Words in tweet')
    plt.show()
    
draw_words_in_tweet(real_tweet_length,false_tweet_length)
real_avg_tweet_len= train_data[train_data['target']==1]['text'].str.split().apply(lambda x:[len(i) for i in x])
real_avg_tweet_len=real_avg_tweet_len.map(lambda x : np.mean(x))
false_avg_tweet_len= train_data[train_data['target']==0]['text'].str.split().apply(lambda x:[len(i) for i in x])
false_avg_tweet_len= false_avg_tweet_len.map(lambda x: np.mean(x))
real_tweet_mention=train_data[train_data['target']==1]['text'].apply(lambda x: len([j for j in str(x) if j=='@']))
false_tweet_mention=train_data[train_data['target']==0]['text'].apply(lambda x : len([j for j in str(x) if j=='@']))
real_stopword_count=train_data[train_data['target']==1]['text'].apply(lambda x: len([j for j in str(x) if j in stop]))
false_stopword_count=train_data[train_data['target']==0]['text'].apply(lambda x: len([j for j in str(x) if j in stop]))
def fig_avg_plot(real_avg_tweet_len,false_avg_tweet_len):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    sns.distplot(real_avg_tweet_len,ax=ax1,color='Blue')
    ax1.set_title('Real Tweet Length')
    sns.distplot(false_avg_tweet_len,ax=ax2,color='Red')
    ax2.set_title('False Tweet Length')
    fig.suptitle('Average Length of Words in Tweet')
    plt.show()


fig_avg_plot(real_avg_tweet_len,false_avg_tweet_len)

def fig_tweet_mention_plot(real_tweet_mention,false_tweet_mention):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    sns.distplot(real_tweet_mention,ax=ax1,color='Blue')
    ax1.set_title('Real Tweet Mention')
    sns.distplot(false_tweet_mention,ax=ax2,color='Red')
    ax2.set_title('False Tweet Mention')
    fig.suptitle('Tweet Mentions')
    plt.show()

print(real_tweet_mention)
print(real_stopword_count)
print(false_stopword_count)
#fig_tweet_mention_plot(real_tweet_mention,false_tweet_mention)
def corpus(data,target):
    corpus_l=[]
    for i in (train_data[train_data['target']==1]['text'].str.split()):
        for j in i:
            corpus_l.append(j)
    return corpus_l


def analyse_most_common(data):
    count=Counter(data)
    mostcommon_words= count.most_common()
    y=[]
    x=[]
    for word,count in mostcommon_words[:150]:
        if word not in stop:
            x.append(word)
            y.append(count)
    
    sns.barplot(x=y,y=x)



def display_cloud(data):
    wordcloud=WordCloud(width=500,height=700,stopwords=stop,background_color='white',min_font_size=5).generate(str(data))
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

real_tweet_corpus= corpus(train_data,1)
false_tweet_corpus=corpus(train_data,0)
#print(false_tweet_corpus)
analyse_most_common(real_tweet_corpus)
analyse_most_common(false_tweet_corpus)
display_cloud(real_tweet_corpus)
display_cloud(false_tweet_corpus)
def remove_url(data):
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data
def clean_data(data):
    emoji_clean= re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    data=emoji_clean.sub(r'',data)
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data
def remove_html(data):
    html_tag=re.compile(r'<.*?>')
    data=html_tag.sub(r'',data)
    return data

def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data


train_data['text']=train_data['text'].apply(lambda x: remove_url(x))
#print(cleaned_data)
train_data['text']=train_data['text'].apply(lambda x: clean_data(x))
print(train_data.head())
train_data['text']=train_data['text'].apply(lambda x: remove_html(x))
train_data['text']=train_data['text'].apply(lambda x: remove_punctuations(x))
#train_data['text']=train_data['text'].apply(lambda x: stem_words(x))
print(train_data.head())



    

    
def vectorize(data):
    cv=CountVectorizer()
    fit_data_cv=cv.fit_transform(data)
    return fit_data_cv,cv
def tfidf(data):
    tfidfv=TfidfVectorizer()
    fit_data_tfidf=tfidfv.fit_transform(data)
    return fit_data_cv,tfidfv

def dimen_reduc_plot(test_data,test_label,option):
    tsvd= TruncatedSVD(n_components=2,algorithm="randomized",random_state=42)
    tsne=TSNE(n_components=2,random_state=42) #not recommended instead use PCA
    pca=SparsePCA(n_components=2,random_state=42)
    if(option==1):
        tsvd_result=tsvd.fit_transform(test_data)
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        
        sns.scatterplot(x=tsvd_result[:,0],y=tsvd_result[:,1],hue=test_label        )
        
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(tsvd_result[:,0],tsvd_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_Tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("TSVD")
        plt.show()
    if(option==2):
        tsne_result=tsne.fit_transform(test_data)
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        sns.scatterplot(x=tsne_result[:,0],y=tsne_result[:,1],hue=test_label)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(x=tsne_result[:,0],y=tsne_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("PCA")
        plt.show() 
    if(option==3):
        pca_result=pca.fit_transform(test_data.toarray())
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        sns.scatterplot(x=pca_result[:,0],y=pca_result[:,1],hue=test_label)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(x=pca_result[:,0],y=pca_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("TSNE")
        plt.show()
        
data_vect=train_data['text'].values
data_vect_real=train_data[train_data['target']==1]['text'].values
target_vect=train_data['target'].values
target_data_vect_real=train_data[train_data['target']==1]['target'].values
data_vect_false=train_data[train_data['target']==0]['text'].values
target_data_vect_false=train_data[train_data['target']==0]['target'].values
train_data_cv,cv= vectorize(data_vect)
real_tweet_train_data_cv,cv=vectorize(data_vect_real)
print(train_data.head())
dimen_reduc_plot(train_data_cv,target_vect,1)
dimen_reduc_plot(real_tweet_train_data_cv,target_data_vect_real,1)
dimen_reduc_plot(train_data_cv,target_vect,3)
dimen_reduc_plot(real_tweet_train_data_cv,target_data_vect_real,3)
dimen_reduc_plot(train_data_cv,target_vect,2)
dimen_reduc_plot(real_tweet_train_data_cv,target_data_vect_real,2)

        
def add_corpus(data):
    corpus=[]
    for i in tqdm(data):
        words=[word.lower() for word in word_tokenize(i)]
        corpus.append(words)
    return corpus
def create_glove_embedding(data):
    embedding_map={}
    file=open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r')
    for  f in file:
        values=f.split(' ')
        word=values[0]
        coef=np.asarray(values[1:],dtype='float32')
        embedding_map[word]=coef
    file.close()
    return embedding_map
def  embedding_preprocess(data,target):
    #max_word_length=1000
    max_sequence_length=100
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences=tokenizer.texts_to_sequences(data)
    
    word_idx=tokenizer.word_index
    data_pad=pad_sequences(sequences,padding="post",maxlen=max_sequence_length)
    label=to_categorical(np.asarray(target))
    print(len(word_idx))
    print("Data Length")
    print(data_pad)
    print("Target Length")
    print(label.shape)
    emb_dim=data.get('a').shape[0]
    print(emb_dim)
    num_length=len(word_idx)+1
    emb_mat=np.zeros((num_length,emb_dim))
    for word,idx in tqdm(word_idx.items()):
        if idx > num_length:
            continue
        elif idx < num_length:
            emb_vector=data.get(word)
            if emb_vector is not None: 
                emb_mat[idx]=emb_vector
    
    return emb_mat,word_idx,data_pad,num_length
    
lines_without_stopwords=[] 
for line in train_data['text'].values: 
    line = line.lower()
    line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE) 
    new_line=[]
    for word in line_by_words:
        if word not in stop:
            new_line.append(word)
    lines_without_stopwords.append(new_line)
texts = lines_without_stopwords
    
corpus_train_data=add_corpus(train_data['text'])
print("corpus created")

targets=train_data['target']
embedding_map= create_glove_embedding(texts)
print("Embedding matrix created")
emb_mat,word_idx,pad_data,num_words=embedding_preprocess(embedding_map,targets)
print(pad_data.shape)

print("Visualise embedded vectors")
plt.plot(emb_mat[10])
plt.plot(emb_mat[20])
plt.plot(emb_mat[50])
plt.title("Embedding Vectors")
plt.show()

def split(train_tweet,test_tweet,random_state,test_size):
    X_train,X_test,Y_train,Y_test= train_test_split(train_tweet,test_tweet,test_size=test_size)
    return X_train,X_test,Y_train,Y_test


print(emb_mat.shape)
print(pad_data.shape)
label=to_categorical(np.asarray(train_data['target']))
print(to_categorical(np.asarray(train_data['target'])).shape)

train_tweet=pad_data[:train_data.shape[0]]
test_tweet=pad_data[train_data.shape[0]:]
print(train_tweet.shape)
print(test_tweet.shape)
X_train,X_test,Y_train,Y_test=split(train_tweet,label,42,0.2)
print(X_train.shape)
print(X_test.shape)
print(train_tweet)


model=Sequential([Embedding(num_words,100,input_length=100,weights=[emb_mat],trainable=False),
                  Conv1D(128,5,activation='relu'),
                  Dropout(0.2),
                  MaxPooling1D(pool_size=3),
                  
                 LSTM(100),
                 Dense(4,activation='relu'),
                  Dense(4,activation='relu'),
                  
                 Dense(2,activation='sigmoid')])
optimizer=Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

model.fit(X_train,Y_train,verbose=2,epochs=40,batch_size=4,validation_data=(X_test,Y_test))
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(test_tweet)
