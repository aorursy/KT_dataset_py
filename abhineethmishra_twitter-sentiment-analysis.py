'''!pip install gensim --upgrade

!pip install keras --upgrade

!pip install pandas --upgrade'''
# DataFrame

import pandas as pd

import gc

# Matplot

import matplotlib.pyplot as plt

%matplotlib inline



# Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM

from keras import utils

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# nltk

import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer



# Word2vec

import gensim



# Utility

import re

import numpy as np

import os

from collections import Counter

import logging

import time

import pickle

import itertools



# Set log

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')
# DATASET

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

DATASET_ENCODING = "ISO-8859-1"

TRAIN_SIZE = 0.8



# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



# WORD2VEC 

W2V_SIZE = 300

W2V_WINDOW = 7

W2V_EPOCH = 32

W2V_MIN_COUNT = 10



# KERAS

SEQUENCE_LENGTH = 300

EPOCHS = 8

BATCH_SIZE = 1024



# SENTIMENT

POSITIVE = "POSITIVE"

NEGATIVE = "NEGATIVE"

NEUTRAL = "NEUTRAL"

SENTIMENT_THRESHOLDS = (0.4, 0.7)



# EXPORT

KERAS_MODEL = "model.h5"

WORD2VEC_MODEL = "model.w2v"

TOKENIZER_MODEL = "tokenizer.pkl"

ENCODER_MODEL = "encoder.pkl"
dataset_filename = os.listdir("../input")[0]

#dataset_path = '../input/sentiment140/training.1600000.processed.noemoticon.csv'

dataset_path = '../input/sentiment140/training.1600000.processed.noemoticon.csv'

print("Open file:", dataset_path)

df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
print("Dataset size:", len(df))
df.head(5)
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

def decode_sentiment(label):

    return decode_map[int(label)]
%%time

df.target = df.target.apply(lambda x: decode_sentiment(x))
target_cnt = Counter(df.target)



plt.figure(figsize=(16,8))

plt.bar(target_cnt.keys(), target_cnt.values())

plt.title("Dataset labels distribuition")
stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)
%%time

df.text = df.text.apply(lambda x: preprocess(x))
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

print("TRAIN size:", len(df_train))

print("TEST size:", len(df_test))
%%time

documents = [_text.split() for _text in df_train.text] 

import os

model_exists = os.path.isfile('../input/trained-model/model.w2v')
if model_exists:

    from gensim.models import Word2Vec

    w2v_model=Word2Vec.load("../input/trained-model/model.w2v")

    print("w2v model exists, existing model loaded.")

else:

    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)
'''if model_present:

    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                                window=W2V_WINDOW, 

                                                min_count=W2V_MIN_COUNT, 

                                                workers=8)'''
if not model_exists:

    w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
w2v_model.wv['model'].shape
%%time

if model_exists==False:

    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

else:

    print("Trained model loaded")
w2v_model.most_similar("love")
%%time

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_train.text)



vocab_size = len(tokenizer.word_index) + 1

print("Total words", vocab_size)
%%time

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=300)

x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=300)
labels = df_train.target.unique().tolist()

labels.append(NEUTRAL)

labels
encoder = LabelEncoder()

encoder.fit(df_train.target.tolist())



y_train = encoder.transform(df_train.target.tolist())

y_test = encoder.transform(df_test.target.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)
print("x_train", x_train.shape)

print("y_train", y_train.shape)

print()

print("x_test", x_test.shape)

print("y_test", y_test.shape)
y_train[:10]
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))

for word, i in tokenizer.word_index.items():

  if word in w2v_model.wv:

    embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=W2V_SIZE, trainable=False)
print(vocab_size,W2V_SIZE,SEQUENCE_LENGTH)
model_exists=os.path.isfile('../input/trained-model/model.h5')

if model_exists:

    from keras.models import load_model

    model = load_model('../input/trained-model/model.h5')

    print("Keras model file found and loaded.")

else:

    model = Sequential()

    model.add(embedding_layer)

    model.add(Dropout(0.5))

    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))



model.summary()
if not model_exists:

    model.compile(loss='binary_crossentropy',

                  optimizer="adam",

                  metrics=['accuracy'])

else:

    print("Compiled Model Loaded.")
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
%%time

if not model_exists:#if model doesn't exist, fit the model on the training set

    from keras.callbacks import CSVLogger



    csv_logger = CSVLogger('training.log', separator=',', append=False)

    history = model.fit(x_train, y_train,

                        batch_size=BATCH_SIZE,

                        epochs=EPOCHS,

                        validation_split=0.1,

                        verbose=1,

                        callbacks=callbacks)

    #storing history for visualization

    pickle_out = open("../input/trained-model/history.pickle","wb")

    pickle.dump(history.histpry, pickle_out)

    pickle_out.close()

else:

    if os.path.isfile('../input/trained-model/history.pickle'):

        history=pickle.load('../input/trained-model/history.pickle')

        print("History file found and loaded.")
%%time

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print()

print("ACCURACY:",score[1])

print("LOSS:",score[0])
try:

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(acc))



    plt.plot(epochs, acc, 'b', label='Training acc')

    plt.plot(epochs, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()

except:

    print("Fit history not found, so visualizations are not shown.")
def decode_sentiment(score, include_neutral=True):

    if include_neutral:        

        label = NEUTRAL

        if score <= SENTIMENT_THRESHOLDS[0]:

            label = NEGATIVE

        elif score >= SENTIMENT_THRESHOLDS[1]:

            label = POSITIVE



        return label

    else:

        return NEGATIVE if score < 0.5 else POSITIVE
def predict(text, include_neutral=True):

    start_at = time.time()

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]

    # Decode sentiment

    label = decode_sentiment(score, include_neutral=include_neutral)



    return {"label": label, "score": float(score),

       "elapsed_time": time.time()-start_at}  
predict("I love the music")
predict("I hate the rain")
predict("i don't know what i'm doing")
%%time

y_pred_1d = []

y_test_1d = list(df_test.target)

scores = model.predict(x_test, verbose=1, batch_size=8000)

y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """



    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=30)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)

    plt.yticks(tick_marks, classes, fontsize=22)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label', fontsize=25)

    plt.xlabel('Predicted label', fontsize=25)
%%time



cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)

plt.figure(figsize=(12,12))

plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")

plt.show()
print(classification_report(y_test_1d, y_pred_1d))
accuracy_score(y_test_1d, y_pred_1d)
model.save(KERAS_MODEL)

w2v_model.save(WORD2VEC_MODEL)

pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)

pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)
predict("lol")
tweets=pd.read_csv('../input/twitterdiamond/diamond.csv')
tweets=tweets.rename(columns={'7:50 pm - 23 May 2019':'Time','This guy told not celebrate the #PulwamaAttack has it hurts  some community  Now people answered him of giving   seat and a defeat of his father and son  Karma hits you  badly  pic twitter com rx eO CK P':'Tweet'})
#!pip install --upgrade pip

import warnings

warnings.filterwarnings("ignore")

tweets.head()
'''pclass=[]

score=[]

count=0

for i in tweets['Tweet']:

    p=predict(i)

    pclass.append(p['label'])

    score.append(p['score'])

    if count%100==0 or count==0:

        print("Tweets done:",count)

    count+=1'''
#tweets["Class"]=pclass

#tweets['Predicted Class Score']=score
tweets.head()
'''from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)'''
#create_download_link(tweets)
#in development



model_exists=os.path.isfile('../input/model-with-dropout/bidirectional_model(with dropout).h5')

if model_exists:

    from keras.models import load_model

    model = load_model('../input/model-with-dropout/bidirectional_model(with dropout).h5')

    print("Keras model file found and loaded.")

else:

    import keras

    from keras.layers import Lambda

    import tensorflow as tf

    model = Sequential()

    model.add(embedding_layer)

    model.add(Dropout(0.5))

    #model.add(Flatten())

    #model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))

    #model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))

    #model.add(keras.layers.Bidirectional((keras.layers.CuDNNLSTM(100,(3,3),padding='same',input_shape=(300,300), dropout=0.2, recurrent_dropout=0.2))))

    model.add(keras.layers.Bidirectional((keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))))

    #model.add(keras.layers.ConvLSTM2D(100, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0))

    model.add(Dense(1, activation='sigmoid'))



model.summary()
predict('The concert was boring for the first 15 minutes while the band warmed up but then was terribly exciting')
if not model_exists:

    model.compile(loss='binary_crossentropy',

                  optimizer="adam",

                  metrics=['accuracy'])

else:

    print("Compiled Model Loaded.")
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
%%time

if not model_exists:#if model doesn't exist, fit the model on the training set

    from keras.callbacks import CSVLogger



    csv_logger = CSVLogger('training.log', separator=',', append=False)

    history = model.fit(x_train, y_train,

                        batch_size=BATCH_SIZE,

                        epochs=EPOCHS,

                        validation_split=0.1,

                        verbose=1,

                        callbacks=callbacks)

    #storing history for visualization

    pickle_out = open("../input/trained-model/convhistory.pickle","wb")

    pickle.dump(history.histpry, pickle_out)

    pickle_out.close()

'''else:

    if os.path.isfile('../input/trained-model/convhistory.pickle'):

        history=pickle.load('../input/trained-model/convhistory.pickle')

        print("History file found and loaded.")'''
model.save('bidirectional_model.h5')
%%time

y_pred_1d = []

y_test_1d = list(df_test.target)

scores = model.predict(x_test, verbose=1, batch_size=8000)

y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """



    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=30)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)

    plt.yticks(tick_marks, classes, fontsize=22)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label', fontsize=25)

    plt.xlabel('Predicted label', fontsize=25)
%%time



cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)

plt.figure(figsize=(12,12))

plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")

plt.show()
print(classification_report(y_test_1d, y_pred_1d))
'''%%time

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print()

print("ACCURACY:",score[1])

print("LOSS:",score[0])'''
# in development

'''

model_exists=os.path.isfile('../input/model-with-dropout/conv_model.h5')

if model_exists:

    from keras.models import load_model

    model = load_model('../input/model-with-dropout/conv_model.h5')

    print("Keras model file found and loaded.")

else:

    import keras

    from keras.layers import Lambda

    import tensorflow as tf

    model = Sequential()

    model.add(embedding_layer)

    model.add(Dropout(0.5))

    #model.add(Flatten())

    model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))

    model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))

    #model.add(keras.layers.Bidirectional((keras.layers.CuDNNLSTM(100,(3,3),padding='same',input_shape=(300,300), dropout=0.2, recurrent_dropout=0.2))))

    model.add(keras.layers.ConvLSTM2D(100,(3,3),padding='same',strides=1, dropout=0.2, recurrent_dropout=0.2,))

    #model.add(keras.layers.ConvLSTM2D(100, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0))

    #model.add(keras.layers.Reshape((300,100)))

    #model.add(Flatten())

    #model.add(Dense(100,activation='sigmoid'))

    model.add(Dense(1,activation='relu'))

    model.add(Flatten())

    model.add(Dense(1,activation='sigmoid'))

    





model.summary()'''
'''# in development



model_exists=os.path.isfile('../input/model-with-dropout/conv_model.h5')

if model_exists:

    from keras.models import load_model

    model = load_model('../input/model-with-dropout/conv_model.h5')

    print("Keras model file found and loaded.")

else:

    import keras

    from keras.layers import Lambda

    import tensorflow as tf

    model = Sequential()

    model.add(embedding_layer)

    model.add(Dropout(0.5))

    #model.add(Flatten())

    #model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))

    #model.add(Lambda(lambda x: tf.expand_dims(model.output, axis=-1)))

    #model.add(keras.layers.Bidirectional((keras.layers.CuDNNLSTM(100,(3,3),padding='same',input_shape=(300,300), dropout=0.2, recurrent_dropout=0.2))))

    #model.add(keras.layers.ConvLSTM2D(100,(3,3),padding='same',strides=1, dropout=0.2, recurrent_dropout=0.2,))

    #model.add(keras.layers.ConvLSTM2D(100, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0))

    model.add(keras.layers.Conv1D(32, 3, strides=1, padding='valid',activation='relu'))

    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid'))

    model.add(keras.layers.Bidirectional((keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))))

    #model.add(keras.layers.Reshape((300,100)))

    #model.add(Flatten())

    #model.add(Dense(100,activation='sigmoid'))

    #model.add(Dense(1,activation='relu'))

    #model.add(Flatten())

    model.add(Dense(1,activation='sigmoid'))

    





model.summary()'''
'''#if not model_exists:

model.compile(loss='binary_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])

#else:

    #print("Compiled Model Loaded.")'''
'''callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]'''
#creating smaller train size

#small_df=pd.concat([pd.DataFrame(x_train).iloc[:10000,:],(pd.DataFrame(y_train).iloc[:100000,:])])
'''x_train=pd.DataFrame(x_train).iloc[:40000,:]

y_train=pd.DataFrame(y_train).iloc[:40000,:]

x_test=pd.DataFrame(x_test).iloc[:7500,:]

y_test=pd.DataFrame(y_test).iloc[:7500,:]

#x_train.shape'''
!export CUDA_VISIBLE_DEVICES=1
'''%%time

#BATCH_SIZE = 16

if not model_exists:#if model doesn't exist, fit the model on the training set

    #from keras.callbacks import CSVLogger



    #csv_logger = CSVLogger('training.log', separator=',', append=False)

    history = model.fit(x_train, y_train,

                        batch_size=1024,

                        epochs=EPOCHS,

                        validation_split=0.1,

                        verbose=1,

                        callbacks=callbacks)

    #storing history for visualization

    pickle_out = open("../convhistory.pickle","wb")

    pickle.dump(history.history, pickle_out)

    pickle_out.close()

    model.save('conv_model.h5')

else:

    if os.path.isfile('../input/trained-model/convhistory.pickle'):

        history=pickle.load('../input/trained-model/convhistory.pickle')

        print("History file found and loaded.")'''
model_exists
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
'''import keras

import tensorflow as tf





config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 

sess = tf.Session(config=config) 

keras.backend.set_session(sess)

init = tf.global_variables_initializer()

sess.run(init)'''
!export CUDA_VISIBLE_DEVICES=1
model_exists
predict("This sucks")
'''%%time

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

print()

print("ACCURACY:",score[1])

print("LOSS:",score[0])'''
df=pd.read_csv('../input/final370data/Article370_Platinum.csv')

df=df.drop(columns='Unnamed: 0')

df.head()
!pip install ray
%%time

#import ray

#ray.init()

#@ray.remote

def getsentiments():

    lst=[]

    for val in df['Text'][150000:]:

        lst.append(predict(val))

    return lst

lst=getsentiments()

#lst=ray.get(getsentiments.remote())
import pickle

with open('SentimentsFrom150k.pickle','wb') as f:

    pickle.dump(lst,f)

'''df['Sentiment']=lst

df.to_csv("a370senti.csv")'''
df.head()
len(lst)
len(df['Text'])