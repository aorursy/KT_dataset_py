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
import numpy as np 
import pandas as pd 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from bs4 import BeautifulSoup
df = pd.read_csv('/kaggle/input/nnfl-lab-4/train.csv')
df_test = pd.read_csv('/kaggle/input/nnfl-lab-4/test.csv')
print(df.shape)
print(df_test.shape)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()
def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = result = re.sub(r"http\S+", "", tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    tweet=re.sub(r'[^\w\s]',' ',tweet, re.UNICODE)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    #to lower case
    tweet=tweet.lower()
    #function to split text into word
    tweet = word_tokenize(tweet)
    #for stop words
    tweet = [w for w in tweet if not w in stop_words]
    #Stemming
    tweet = [lemmatizer.lemmatize(token) for token in tweet]
    tweet = [lemmatizer.lemmatize(token, "v") for token in tweet]
    tweet = " ".join(tweet)


    return tweet

def apply_on_data(data):
    temp_data=data.copy()
    data_clean1 = [clean_tweet(tweet) for tweet in data.Sentence1]
    df_temp1=pd.DataFrame(data_clean1,columns=["tweet"])
    temp_data["Sentence1"]=df_temp1["tweet"]

    data_clean2 = [clean_tweet(tweet) for tweet in data.Sentence2]
    df_temp2=pd.DataFrame(data_clean2,columns=["tweet"])
    temp_data["Sentence2"]=df_temp2["tweet"]

    return temp_data
df=apply_on_data(df)
test_clean=apply_on_data(df_test)
df.head()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , CuDNNLSTM , Embedding, Dropout , Activation, GRU, Flatten ,LSTM,Conv1D,BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPooling1D, concatenate, dot,subtract,MaxPooling1D,multiply,GlobalAveragePooling1D
from keras.models import Model, Sequential
import keras.backend as K
total_text = pd.concat([df['Sentence1'], df['Sentence2']]).reset_index(drop=True)
num_words = 5000
tokenizer = Tokenizer(num_words=num_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
tokenizer.fit_on_texts(total_text)
sentence_1_sequenced = tokenizer.texts_to_sequences(df['Sentence1'])
sentence_2_sequenced = tokenizer.texts_to_sequences(df['Sentence2'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print(word_index)
print(df["Sentence1"][0])
print(sentence_1_sequenced[0])
max_length_of_text = 100

sentence_1_sequenced = pad_sequences(sentence_1_sequenced, maxlen=max_length_of_text)
sentence_2_sequenced = pad_sequences(sentence_2_sequenced, maxlen=max_length_of_text)
total_text_test=pd.concat([test_clean['Sentence1'], test_clean['Sentence2']]).reset_index(drop=True)
tokenizer.fit_on_texts(total_text)
sentence_1_sequenced_test = tokenizer.texts_to_sequences(test_clean['Sentence1'])
sentence_2_sequenced_test = tokenizer.texts_to_sequences(test_clean['Sentence2'])

sentence_1_sequenced_test = pad_sequences(sentence_1_sequenced_test, maxlen=max_length_of_text)
sentence_2_sequenced_test = pad_sequences(sentence_2_sequenced_test, maxlen=max_length_of_text)
y=df["Class"]
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
earlystop = EarlyStopping(patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
checkpointer = ModelCheckpoint(filepath='weights1.h5', verbose=1, monitor='val_accuracy',save_weights_only=True,save_best_only=True)
callbacks = [earlystop, learning_rate_reduction,checkpointer]
embeddings_index = {}
f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float64')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors in pretrained word vector model.' % len(embeddings_index))
print('Dimensions of the vector space : ', len(embeddings_index['the']))
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding,Lambda

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_length_of_text,
                            trainable=False)
def model_conv1D_(emb_matrix):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=max_length_of_text,
        trainable=False
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(max_length_of_text,))
    seq2 = Input(shape=(max_length_of_text,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
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

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # Add the magic features
    magic_input = Input(shape=(5,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    distance_input = Input(shape=(20,))
    distance_dense = BatchNormalization()(distance_input)
    distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer
    merge = concatenate([diff, mul])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
Model1=model_conv1D_(embedding_matrix)
Model1.summary()
batch_size = 20
epochs = 10
Model1.fit([sentence_1_sequenced, sentence_2_sequenced], y,batch_size=20, epochs=epochs, validation_split=0.08,callbacks=callbacks)
Model1.load_weights("weights1.h5")
prediction=Model1.predict([sentence_1_sequenced_test,sentence_2_sequenced_test])
prediction=np.round(prediction).astype(int)
df_final=pd.DataFrame(prediction,columns=["Class"])
df_final["Class"].value_counts()
sample1=pd.read_csv("/kaggle/input/nnfl-lab-4/sample_submission.csv")
sample1["Class"]=df_final["Class"]
df=sample1
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
