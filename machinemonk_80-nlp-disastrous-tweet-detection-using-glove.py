import numpy as np # linear algebra

from numpy.random import seed

seed(1)

import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt



import re

import nltk

from collections import defaultdict

from collections import  Counter

import seaborn as sns



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

tweet = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

testset = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

tweet.head()
ax = tweet.replace({"target" : {1 : "Disaster", 0 : "Not disaster"}}).groupby(['target'])['target'].count().plot.bar(title = "Train set count by disaster/not disaster")

_ = ax.set_xlabel('Disaster?')

_ = ax.set_ylabel('Count')

def create_corpus():

    corpus=[]

    

    for x in tweet['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus = create_corpus()

lst_stopwords = nltk.corpus.stopwords.words("english")



counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in lst_stopwords) :

        x.append(word)

        y.append(count)

        

sns.barplot(x=y,y=x)
def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_tweet_bigrams=get_top_tweet_bigrams(corpus)[:10]

x,y=map(list,zip(*top_tweet_bigrams))

sns.barplot(x=y,y=x)
import operator



def build_vocab(X):

    

    tweets = X.apply(lambda s: s.split()).values      

    vocab = {}

    

    for tweet in tweets:

        for word in tweet:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1                

    return vocab



def check_embeddings_coverage(X, embeddings):

    

    vocab = build_vocab(X)    

    

    covered = {}

    oov = {}    

    n_covered = 0

    n_oov = 0

    

    for word in vocab:

        try:

            covered[word] = embeddings[word]

            n_covered += vocab[word]

        except:

            oov[word] = vocab[word]

            n_oov += vocab[word]

            

    vocab_coverage = len(covered) / len(vocab)

    text_coverage = (n_covered / (n_covered + n_oov))

    

    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_oov, vocab_coverage, text_coverage
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
train_glove_oov, train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(tweet["text"], embedding_dict)

test_glove_oov, test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(testset['text'], embedding_dict)

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))

def utils_preprocess_text(text):

        ## clean (convert to lowercase and remove punctuations and characters and then strip)

    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

     

    ## clean urls 

    url = re.compile(r'https?://\S+|www\.\S+')

    text = url.sub(r'',text)

    

    url = re.compile(r'http?://\S+|www\.\S+')

    text = url.sub(r'',text)

    

    ## remove html 

    html=re.compile(r'<.*?>') 

    html.sub(r'',text)

        

        

    text = re.sub(r'mh370','flight crash',text)     

    text = re.sub(r'û_','',text)     

    text = re.sub(r'ûò','',text) 

    text = re.sub(r'typhoondevastated','typhoon devastated',text)      

    text = re.sub(r'irandeal','iran deal',text)      

    text = re.sub(r'worldnews','world news',text)      

    text = re.sub(r'animalrescue','animal rescue',text)      

    text = re.sub(r'viralspell','viral spell',text)      

    text = re.sub(r'griefûª','grief',text)      

    text = re.sub(r'pantherattack','panther attack',text)      

    text = re.sub(r'injuryi495','injury in 495',text) 

    text = re.sub(r'explosionproof','explosion proof',text) 

    text = re.sub(r'americaûªs','americans',text) 

    return text

df=pd.concat([tweet,testset])

df.shape


df["text_clean"] = df["text"].apply(lambda x: utils_preprocess_text(x))

df.head()
df_glove_oov, df_glove_vocab_coverage, df_glove_text_coverage = check_embeddings_coverage(df["text_clean"], embedding_dict)

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(df_glove_vocab_coverage, df_glove_text_coverage))

corpus = []

corpus = df["text_clean"]
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i < num_words:

        emb_vec=embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec   
model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=3e-4)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
train=tweet_pad[:tweet.shape[0]]

test=tweet_pad[tweet.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.2)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test),verbose=2)
train_pred_GloVe = model.predict(train)

train_pred_GloVe_int = train_pred_GloVe.round().astype('int')
test_pred_GloVe = model.predict(test)

test_pred_GloVe_int = test_pred_GloVe.round().astype('int')



submission['target'] = test_pred_GloVe_int

submission.head(10)



submission.to_csv("submission.csv", index=False, header=True)
