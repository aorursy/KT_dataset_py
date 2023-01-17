import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from collections import Counter

from plotly import graph_objs as go
from sklearn import preprocessing 
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout, Bidirectional, Conv2D
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer
from keras.layers import LSTM,Dense,Bidirectional,Input
from keras.models import Model
import torch
import transformers
df = pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/data.csv")
df.head()
df.columns
# Adding the title and body of query
df['text'] = df['Title'] + " " + df['Body']

# Drop columns not used for modelling
cols_to_drop = ['Id', 'Tags', 'CreationDate', 'Title', 'Body']
df.drop(cols_to_drop, axis=1, inplace=True)

# Rename category column to be more meaningful
df = df.rename(columns={"Y": "class"})

print("Total number of samples:", len(df))

df.head()
temp = df.groupby('class').count()['text'].reset_index()

fig = go.Figure(go.Funnelarea(
    text = temp['class'],
    values = temp.text,
    title = {"position" : "top center", "text" : "Funnel Chart for target distribution"}
    ))
fig.show()
temp = df['class'].value_counts()

fig = px.bar(temp)
fig.update_layout(
    title_text='Data distribution for each category',
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Label'
    )
)
fig.show()
high = df[df['class']=='HQ']['text'].str.split().map(lambda x: len(x) if len(x) < 500 else 500)
low_open = df[df['class']=='LQ_EDIT']['text'].str.split().map(lambda x: len(x) if len(x) < 500 else 500)
low_closed = df[df['class']=='LQ_CLOSE']['text'].str.split().map(lambda x: len(x) if len(x) < 500 else 500)

fig = go.Figure()
fig.add_trace(go.Histogram(x=high, histfunc='avg', name="HQ", opacity=0.6, histnorm='probability density'))
fig.add_trace(go.Histogram(x=low_open, histfunc='avg', name="LQ_EDIT", opacity=0.6, histnorm='probability density'))
fig.add_trace(go.Histogram(x=low_closed, histfunc='avg', name="LQ_CLOSE", opacity=0.6, histnorm='probability density'))

fig.update_layout(
    title_text='Number of words in post', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2,
    bargroupgap=0.1,
    barmode='overlay'
)
fig.show()
df.isna().sum() # Checking for nan Values
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
# Clean the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    return text
df['text'] = df['text'].apply(clean_text)
plt.figure(figsize = (20,20)) # Text that is of high quality
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df['class'] == 'HQ'].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is of low quality(closed)
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df['class'] == 'LQ_EDIT'].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is of low quality(open)
wc = WordCloud(max_words = 1000 , width = 1400 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df['class'] == 'LQ_CLOSE'].text))
plt.imshow(wc , interpolation = 'bilinear')
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'class'. 
df['class']= label_encoder.fit_transform(df['class']) 
  
df['class'].unique() 
df['temp_list'] = df['text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in df['temp_list'].loc[df['class'] == 0] for item in sublist])
top_hq = pd.DataFrame(top.most_common(15))
top_hq.columns = ['Common_words','count']

fig = px.bar(top_hq, x='count',y='Common_words',title='Common words in High Quality posts',orientation='h',width=700,height=500,color='Common_words')
fig.show()

fig = px.treemap(top_hq, path=['Common_words'], values='count',title='Tree of Common words in High Quality posts')
fig.show()
fig = px.pie(top_hq,
             values='count',
             names='Common_words',
             title='Word distribution in High Quality posts')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
top = Counter([item for sublist in df['temp_list'].loc[df['class'] == 1] for item in sublist])
top_lq = pd.DataFrame(top.most_common(15))
top_lq.columns = ['Common_words','count']

fig = px.bar(top_lq, x='count',y='Common_words',title='Common words in Low Quality posts(Closed)',orientation='h',width=700,height=500,color='Common_words')
fig.show()

fig = px.treemap(top_lq, path=['Common_words'], values='count',title='Tree of Common words in Low Quality posts')
fig.show()
fig = px.pie(top_lq,
             values='count',
             names='Common_words',
             title='Word distribution in Low Quality posts(Closed)')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
top = Counter([item for sublist in df['temp_list'].loc[df['class'] == 2] for item in sublist])
top_lq = pd.DataFrame(top.most_common(15))
top_lq.columns = ['Common_words','count']

fig = px.bar(top_lq, x='count',y='Common_words',title='Common words in Low Quality posts(Open)',orientation='h',width=700,height=500,color='Common_words')
fig.show()

fig = px.treemap(top_lq, path=['Common_words'], values='count',title='Tree of Common words in Low Quality posts')
fig.show()
del df['temp_list']
fig = px.pie(top_lq,
             values='count',
             names='Common_words',
             title='Word distribution in Low Quality posts(Open)')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
df.head()
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(16,6))
text_len=df[df['class']==0]['text'].str.len()
ax1.hist(text_len,color='lightcoral')
ax1.set_title('High Quality')
text_len=df[df['class']==1]['text'].str.len()
ax2.hist(text_len,color='lightgreen')
ax2.set_title('Low Quality(closed)')
text_len=df[df['class']==2]['text'].str.len()
ax3.hist(text_len,color='lightskyblue')
ax3.set_title('Low Quality(open)')
fig.suptitle('Characters in texts')
plt.show()
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(16,6))
word=df[df['class']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='lightcoral')
ax1.set_title('High Quality')
word=df[df['class']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='lightgreen')
ax2.set_title('Low Quality(closed)')
word=df[df['class']==2]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='lightskyblue')
ax3.set_title('Low Quality(open)')
fig.suptitle('Average word length in each text')
df.head()
from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
plt.figure(figsize = (16,5))
most_common_uni = get_top_text_ngrams(df.text,10,1)
most_common_uni = dict(most_common_uni)
sns.set_palette("husl")
sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))
plt.figure(figsize = (16,5))
most_common_bi = get_top_text_ngrams(df.text,10,2)
most_common_bi = dict(most_common_bi)
sns.set_palette("husl")
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))
plt.figure(figsize = (16,5))
most_common_tri = get_top_text_ngrams(df.text,10,3)
most_common_tri = dict(most_common_tri)
sns.set_palette("husl")
sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))
df.head()
x_train,x_test,y_train,y_test = train_test_split(df['text'], df['class'], test_size = 0.2, random_state = 42, stratify = df['class'])
max_features = 10000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)
tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(x_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
EMBEDDING_FILE = '../input/glove-twitter/glove.twitter.27B.200d.txt'
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
#change below line if computing normal stats is too slow
embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
batch_size = 256
epochs = 5
embed_size = 200
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.4, min_lr=0.0000001)
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    #Defining Neural Network
    model = Sequential()
    #Non-trainable embeddidng layer
    model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=True))
    #LSTM 
    model.add(Bidirectional(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.4 , dropout = 0.4)))
    model.add(Bidirectional(LSTM(units=128 , recurrent_dropout = 0.2 , dropout = 0.2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr = 0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , epochs = epochs , callbacks = [learning_rate_reduction])
print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
epochs = [i for i in range(5)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
pred = model.predict_classes(X_test)
print(classification_report(y_test, pred, target_names = ['HQ', 'LQ(Close)', 'LQ(Open)']))
cm = confusion_matrix(y_test,pred)
cm
cm = pd.DataFrame(cm , index = ['HQ', 'LQ(Close)', 'LQ(Open)'] , columns = ['HQ', 'LQ(Close)', 'LQ(Open)'])

plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['HQ', 'LQ(Close)', 'LQ(Open)'] , yticklabels = ['HQ', 'LQ(Close)', 'LQ(Open)'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
x_train,x_test,y_train,y_test = train_test_split(df['text'], df['class'], random_state = 0 , stratify = df['class'])
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased' , lower = True)
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)
fast_tokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=200):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
x_train = fast_encode(x_train.values, fast_tokenizer, maxlen=200)
x_test = fast_encode(x_test.values, fast_tokenizer, maxlen=200)
# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    max_len=200
    transformer =  transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(3, activation='softmax')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(lr=7e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train,y_train,batch_size = 64 ,validation_data=(x_test,y_test),epochs = 5)
print("Accuracy of the model on Testing Data is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
epochs = [i for i in range(5)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
pred = model.predict(x_test)
cm = pd.DataFrame(cm , index = ['HQ', 'LQ(Close)', 'LQ(Open)'] , columns = ['HQ', 'LQ(Close)', 'LQ(Open)'])

plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['HQ', 'LQ(Close)', 'LQ(Open)'] , yticklabels = ['HQ', 'LQ(Close)', 'LQ(Open)'])
plt.xlabel("Predicted")
plt.ylabel("Actual")