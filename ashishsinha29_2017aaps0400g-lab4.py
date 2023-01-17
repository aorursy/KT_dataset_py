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
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model,Sequential
from keras.layers import Dense, Embedding,TimeDistributed, Input,BatchNormalization,CuDNNGRU,MaxPool1D,GlobalAveragePooling1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,Conv1D,GRU,Flatten,merge,Lambda,merge,Add
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from keras.layers.merge import concatenate
import keras.backend as K
train = pd.read_csv('/kaggle/input/nnfl-lab-4/train.csv')
test = pd.read_csv('/kaggle/input/nnfl-lab-4/test.csv')
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text
train['Sentence1']=train['Sentence1'].apply(preprocessor)
train['Sentence2']=train['Sentence2'].apply(preprocessor)
test['Sentence1']=test['Sentence1'].apply(preprocessor)
test['Sentence2']=test['Sentence2'].apply(preprocessor)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
stop=stopwords.words('english')
lemmatizer = WordNetLemmatizer() 
porter=PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
def remove_stopwords(text):
    return [w for w in text if w not in stop]
def combine(text):
    return  ' '.join(text)
train['Sentence1']=train['Sentence1'].apply(tokenizer_porter).apply(remove_stopwords).apply(combine)
train['Sentence2']=train['Sentence2'].apply(tokenizer_porter).apply(remove_stopwords).apply(combine)
test['Sentence1']=test['Sentence1'].apply(tokenizer_porter).apply(remove_stopwords).apply(combine)
test['Sentence2']=test['Sentence2'].apply(tokenizer_porter).apply(remove_stopwords).apply(combine)
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 5000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 128
EPOCHS = 20
word2vec={}
with open(os.path.join('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.%sd.txt' % EMBEDDING_DIM),encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
    f.close()
print ('found %s word vectors.' %len(word2vec))     
sentences = train['Sentence1'].values
sentences = np.append(sentences,(train['Sentence2'].values))
sentences = np.append(sentences,(test['Sentence1'].values))
sentences = np.append(sentences,(test['Sentence2'].values))
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences1 = tokenizer.texts_to_sequences(train['Sentence1'].values)
sequences2 =  tokenizer.texts_to_sequences(train['Sentence2'].values)
test_sequences1 = tokenizer.texts_to_sequences(test['Sentence1'].values)
test_sequences2 =  tokenizer.texts_to_sequences(test['Sentence2'].values)
word2idx = tokenizer.word_index

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)

data1.shape,data2.shape
test_data1 = pad_sequences(test_sequences1, maxlen=MAX_SEQUENCE_LENGTH)
test_data2 = pad_sequences(test_sequences2, maxlen=MAX_SEQUENCE_LENGTH)
test_data1.shape,test_data2.shape
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
    if i<MAX_VOCAB_SIZE:
        embedding_vector= word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
embedding_layer=Embedding(
num_words,
EMBEDDING_DIM,
weights=[embedding_matrix],
input_length=MAX_SEQUENCE_LENGTH,
trainable=True)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


file_path = "best_model.h5"
# ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
#                         save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks = [early, learning_rate_reduction]
y=train['Class']
emb_layer = Embedding(
num_words,
EMBEDDING_DIM,
weights=[embedding_matrix],
input_length=MAX_SEQUENCE_LENGTH,
trainable=True)

conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

seq1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
seq2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

emb1 = emb_layer(seq1)
emb2 = emb_layer(seq2)

conv1a = conv1(emb1)
glob1a = GlobalAveragePooling1D()(conv1a)
conv1b = conv1(emb2)
glob1b = GlobalAveragePooling1D()(conv1b)

conv2a = conv2(emb1)
glob2a = GlobalAveragePooling1D()(conv2a)
conv2b = conv2(emb2)
glob2b = GlobalAveragePooling1D()(conv2b)

conv3a = conv3(emb1)
glob3a = GlobalAveragePooling1D()(conv3a)
conv3b = conv3(emb2)
glob3b = GlobalAveragePooling1D()(conv3b)

conv4a = conv4(emb1)
glob4a = GlobalAveragePooling1D()(conv4a)
conv4b = conv4(emb2)
glob4b = GlobalAveragePooling1D()(conv4b)

conv5a = conv5(emb1)
glob5a = GlobalAveragePooling1D()(conv5a)
conv5b = conv5(emb2)
glob5b = GlobalAveragePooling1D()(conv5b)

conv6a = conv6(emb1)
glob6a = GlobalAveragePooling1D()(conv6a)
conv6b = conv6(emb2)
glob6b = GlobalAveragePooling1D()(conv6b)

mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

merge = concatenate([diff, mul])

x = Dropout(0.2)(merge)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)

x = Dropout(0.2)(x)
x = BatchNormalization()(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[seq1, seq2], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit([data1,data2], y=y, batch_size=1000,epochs=EPOCHS,verbose=1,validation_split=0.1, shuffle=True)
sub=pd.read_csv('/kaggle/input/nnfl-lab-4/sample_submission.csv')
pred=model.predict([test_data1,test_data2])
for i in range(len(sub)):
    sub.iloc[i,1]=1 if pred[i][0]>0.5 else 0
df = sub

from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)
