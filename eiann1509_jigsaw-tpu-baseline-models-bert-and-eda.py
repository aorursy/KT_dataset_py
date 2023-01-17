import tensorflow as tf 

import pandas as pd

import numpy as np
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
df=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

df.head()
df=df[:1000]
df.info() #checking missing data and datype in columns
df=df.drop(columns=['severe_toxic','obscene','threat','insult','identity_hate'])
y=df['toxic'].values   #target data
#cleaning the text in train data

import re



def clean_text(data):

   x=re.sub(r'https?://\S+|www\.\S+','',data) #hyperlinks

   x=re.sub(r'[!@#$"]','',x) #symbols

   x=re.sub(r'\d+','', x) #digits

   return x

X=[clean_text(i) for i in df['comment_text']]

X=np.array(X)   

X[5]
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords



t_x=[]

STOPWORDS=stopwords.words('english')

z=[]

for i in X:

  for word in i.split():

    if word not in STOPWORDS:

      z+=word

      z+=' '

  t_x.append(''.join(z))

  z=[]
feature_x=np.array(t_x) #required feature data
feature_x[0:3]
feature_x.shape,y.shape
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(feature_x,y,test_size=0.3,random_state=32)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer



tfidf=TfidfVectorizer(strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



table_tf=tfidf.fit(list(x_train)+list(x_test))

table_tf1=tfidf.transform(x_train)

table_tf2=tfidf.transform(x_test)
# it is not used but its performance can also be checked by replacing tfidfvectorizer

"""vec=CountVectorizer(analyzer='word',token_pattern=r'\w{1,}', #4 gram 

            ngram_range=(1, 4), stop_words = 'english')



table_c=vec.fit_transform(list(x_train)+list(x_test))

table_c1=vec.transform(x_train)

table_c2=vec.transform(x_test)"""
table_tf1.shape,y_train.shape
from sklearn.linear_model import LogisticRegression 



classifier=LogisticRegression()

classifier.fit(table_tf1,y_train)

preds=classifier.predict_proba(table_tf2)

preds[:1]
from sklearn.metrics import log_loss  #loss value

loss=log_loss(y_test, preds)

print(loss)
from sklearn.metrics import roc_auc_score    #roc_auc_score

score=roc_auc_score(y_test,preds.argmax(axis=1))

score
#applying SVM



from sklearn.svm import SVC

classifier_1=SVC(probability=True)

classifier_1.fit(table_tf1,y_train)

preds_2=classifier_1.predict_proba(table_tf2)
loss_2=log_loss(y_test,preds_2)  #loss value

print(loss_2)
from sklearn.metrics import roc_auc_score   #roc_auc_score

score=roc_auc_score(y_test,preds_2.argmax(axis=1))

score
#applying decision tree



from sklearn.tree import DecisionTreeClassifier



classifier_2=DecisionTreeClassifier()

classifier_2.fit(table_tf1,y_train)

preds_3=classifier_2.predict_proba(table_tf2)
loss_3=log_loss(y_test,preds_2) #loss value

print(loss_3)
from sklearn.metrics import roc_auc_score   #roc_auc_score

score=roc_auc_score(y_test,preds_3.argmax(axis=1))

score
#keras model using pre-trained glove embeddings
import tensorflow as tf



from tensorflow.keras.layers import Input,Dense,LSTM,SpatialDropout1D,Bidirectional,Dropout,TimeDistributed,Embedding

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.regularizers import Regularizer

tokenizer=Tokenizer()

tokenizer.fit_on_texts(x_train)

sequence=tokenizer.texts_to_sequences(x_train)



word_index=tokenizer.word_index

vocab_len=len(tokenizer.word_index)

max_len=max([len(i) for i in sequence])



sequences=pad_sequences(sequence,maxlen=max_len)      #train_sequences

#preprocessing of test_sequences

test=[]                  

alpha=[]



for x in x_test: 

   for i in x.split():

     if i in word_index.keys():

       test.append(i)

   alpha.append(' '.join(test))  

   test=[]
test_set=[]

test_set_y=[]

for i in range(len(alpha)):

   if alpha[i]!='':

     test_set.append(alpha[i])

     test_set_y.append(y_test[i])

test_set[:2],test_set_y[:2]
test_set_y=np.array(test_set_y)

test_set_y.shape  #target test_sequences
test_set=tokenizer.texts_to_sequences(test_set)



sequence_test=pad_sequences(test_set,maxlen=max_len) #test_sequences
sequence_test.shape


#using transfer learning

#loading pre-trained glove model



!wget https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt
embedding_dim=100

embedding_index = {};

with open('./glove.6B.100d.txt') as f:

    for line in f:

        values = line.split();

        word = values[0];

        coefs = np.asarray(values[1:], dtype='float32');

        embedding_index[word] = coefs;



embedding_mat = np.zeros((vocab_len+1, embedding_dim));

for word, i in word_index.items():

    if word in list(embedding_index.keys()):

      if i!=7581:

        embedding_mat[i]=embedding_index.get(word)

  

sequences.shape,sequence_test.shape,y_train.shape,test_set_y.shape
from tensorflow.keras.optimizers import Adam



#building model



i=Input(shape=(819,))

x=Embedding(vocab_len+1,embedding_dim,weights=[embedding_mat],trainable=False)(i)

x=Bidirectional(LSTM(512))(x)

x=Dropout(0.2)(x)

x=Dense(256,activation='relu')(x)

x=Dropout(0.2)(x)

x=Dense(1,activation='sigmoid')(x)



model=Model(i,x)



model.compile(optimizer=Adam(0.001),loss='binary_crossentropy',metrics=['binary_accuracy'])
model.summary()
r=model.fit(sequences,y_train,validation_data=(sequence_test,test_set_y),epochs=3,verbose=1)
import matplotlib.pyplot as plt

plt.plot(r.history['loss'],label='loss')

plt.plot(r.history['val_loss'],label='val_loss')

plt.legend()
import transformers

from transformers import AutoModel, AutoTokenizer, BertTokenizer

from tokenizers import BertWordPieceTokenizer

from transformers import AutoModel, AutoTokenizer, BertTokenizer
import pandas as pd

train_x1=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

train_x2=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')

test_d=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')

val_d=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
train_x1=train_x1[['comment_text','toxic']]

train_x1.head(1)
train_x2=train_x2[['comment_text','toxic']]

train_x2.head(1)
len(train_x2[train_x2['toxic']==1]) #checking
final_train_d=pd.concat([train_x1,train_x2[train_x2['toxic']==1],train_x2[train_x2['toxic']==0]])
final_train_d=final_train_d[:1000]

final_train_d.head(1)
test_d=test_d[['content','lang']]

test_d=test_d[:1000]

test_d.head(3)
val_d=val_d[['comment_text','lang','toxic']]

val_d=val_d[:1000]

val_d.head(3)
import tensorflow as tf

#IMP DATA FOR CONFIG



AUTO = tf.data.experimental.AUTOTUNE





# Configuration

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192
"""https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras"""



def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):



    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

   

    all_ids = []

    

    for i in range(0, len(texts), chunk_size):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
tokenizer=transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

tokenizer.save_pretrained('.')



fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

fast_tokenizer
x_train_d=fast_encode(final_train_d['comment_text'].astype(str),fast_tokenizer,maxlen=MAX_LEN)

x_valid_d=fast_encode(val_d['comment_text'].astype(str),fast_tokenizer,maxlen=MAX_LEN)

x_test_d=fast_encode(test_d['content'].astype(str),fast_tokenizer,maxlen=MAX_LEN)

type(x_train_d)
y_train_d=final_train_d['toxic'].values

y_valid_d=val_d['toxic'].values

y_test_d=test_d['lang'].values
train_dataset=(tf.data.Dataset

              .from_tensor_slices((x_train_d,y_train_d))

              .repeat()

              .shuffle(2048)

              .batch(BATCH_SIZE)

              .cache()

              .prefetch(AUTO))



valid_dataset = (tf.data.Dataset

    .from_tensor_slices((x_valid_d, y_valid_d))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test_d)

    .batch(BATCH_SIZE)

)
from tensorflow.keras.optimizers import Adam

#building_model



def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
with strategy.scope():

    transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-multilingual-cased')

    )

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train_d.shape[0] // 100

r=model.fit(train_dataset,steps_per_epoch=n_steps,validation_data=valid_dataset,epochs=5)
plt.plot(r.history['loss'],label='loss')

plt.plot(r.history['accuracy'],label='accuracy')



plt.legend()