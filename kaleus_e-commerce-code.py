!python -m spacy download pt

!pip install unidecode

!pip install spacymoji
import pandas as pd

import numpy as np

import spacy

from tqdm import tqdm

import nltk

import keras

import math

import matplotlib

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from math import sqrt

import re

import string

import unidecode

import unicodedata



from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, LSTM,concatenate, SpatialDropout1D, GlobalAveragePooling1D, Input, Bidirectional, MaxPooling1D, Activation,Conv1D,GRU, CuDNNGRU, CuDNNLSTM, Dropout, GlobalMaxPooling1D,Embedding

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

import tensorflow as tf

from keras import backend as K

from keras import backend



from sklearn.model_selection import ShuffleSplit

import itertools



tqdm.pandas()

%matplotlib inline
# Create folder to save embeddings

!mkdir word_embeddings/
#Embeddings download



#DOWNLOAD GLOVE

!wget -c http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip -O glove_s300.zip

!unzip glove_s300.zip

!mv glove_s300.txt word_embeddings/g.txt

!rm glove_s300.zip



#DOWNLOAD WORD2VEC

#!wget -c http://143.107.183.175:22980/download.php?file=embeddings/word2vec/cbow_s300.zip -O word_s300.zip

#!unzip word_s300.zip

#!mv cbow_s300.txt word_embeddings/w.txt



#DOWNLOAD FAST

#!wget -c http://143.107.183.175:22980/download.php?file=embeddings/fasttext/skip_s300.zip -O fast_s300.zip

#!unzip fast_s300.zip

#!mv skip_s300.txt word_embeddings/f.txt
#Load Data



train_df = pd.read_csv('/kaggle/input/e-commerce-reviews/train.csv')

test_df = pd.read_csv('/kaggle/input/e-commerce-reviews/test.csv')
# Score count

train_df['review_score'].value_counts()
train_df.groupby('review_score').describe()
train_df.info()
# Look for null values

train_df.isnull().sum()
# List rows containing null columns

train_df[train_df.isna().any(axis=1)]
# Remove null values

train_df = train_df.dropna()



# New shape

train_df.shape
# Count words that appears most in the text

qtd_words = pd.Series(' '.join(train_df['review_comment_message']).lower().split()).value_counts()

qtd_words[:10]
# Transcription given for each one emojis on spacy, and his translation

emojis = {'smiling face with heart-eyes' : 'amei',

 'grinning face with big eyes' : 'legal',

 'clapping hands' : 'palmas',

 'beaming face with smiling eyes' : 'legal',

 'disappointed face' : 'desapontado',

 'thumbs up medium-light skin tone' : 'gostei',

 'clapping hands medium-dark skin tone' : 'palmas',

 'elephant' : 'elefante',

 'OK hand medium-light skin tone' : 'ok',

 'thumbs up' : 'gostei',

 'clapping hands light skin tone' : 'palmas',

 'thumbs up light skin tone' : 'gostei',

 'hugging face' : 'gostei',

 'tulip' : 'flor',

 'call me hand light skin tone' : 'legal',

 'thumbs down' : 'não gostei',

 'grinning face' : 'feliz',

 'thinking face' : 'pensativo',

 'dog' : 'cachorro',

 'kiss mark' : 'beijo',

 'two hearts' : 'coração',

 'revolving hearts' : 'coração',

 'face blowing a kiss' : 'beijo',

 'green heart' : 'coração verde',

 'smiling face with smiling eyes' : 'legal',

 'thumbs up medium skin tone' : 'gostei',

 'heart decoration' : 'coração',

 'sparkling heart': 'coração',

 'heart with ribbon': 'coração',

 'angry face' : 'raiva',

 'folded hands' : 'obrigado',

 'clapping hands medium skin tone' : 'palmas',

 'persevering face' : 'perseverante',

 'neutral face' : 'neutro',

 'pouting face' : 'raiva',

 'glowing star' : 'estrela',

 'winking face' : 'piscar',

 'pensive face' : 'pensativo',

 'hundred points' : 'cem pontos',

 'grinning squinting face' : 'legal',

 'smirking face' : 'sorrindo',

 'girl' : 'garota',

 'backhand index pointing up' : 'acima',

 'backhand index pointing right' : 'na frente',

 'loudly crying face' : 'chorando',

 'sad but relieved face' : 'preocupado',

 'crying face' : 'chorando',

 'clapping hands medium-light skin tone' : 'palmas',

 'television' : 'televisão',

 'smiling face with sunglasses' : 'sorrindo',

 'drooling face' : 'água na boca',

 'weary face' : 'cansado',

 'oncoming fist' : 'punho',

 'confused face' : 'confuso',

 'thumbs down light skin tone' : 'não gostei',

 'beating heart' : 'coração',

 'horse racing' : 'cavalo',

 'wrapped gift' : 'presente',

 'downcast face with sweat' : 'desanimado',

 'TOP arrow' : 'top',

 'slightly frowning face' : 'desanimado',

 'four leaf clover' : 'trevo',

 'oncoming fist medium skin tone' : 'punho',

 'OK hand light skin tone' : 'ok'}
import spacy

from spacymoji import Emoji



nlp = spacy.load('pt')

emoji = Emoji(nlp)

nlp.add_pipe(emoji, first=True)



def replace_emoji(x):

  doc = nlp(x)

  words = []

  if doc._.has_emoji:

    for index, item in enumerate(doc):

      emo_ind = 0

      if doc[index]._.is_emoji:

        words.append(doc._.emoji[emo_ind][2])

        emo_ind = emo_ind + 1

      else:

        words.append(doc[index].text)

    return ' '.join(words)

      

  else:

    return x
def translate_emoji(x):

  for item in emojis:

    x = x.replace(item, emojis[item])

  return x
#replace emoji

train_df['review_comment_message'] = train_df['review_comment_message'].progress_map(replace_emoji)

train_df['review_comment_message'] = train_df['review_comment_message'].progress_map(translate_emoji)



test_df['review_comment_message'] = test_df['review_comment_message'].progress_map(replace_emoji)

test_df['review_comment_message'] = test_df['review_comment_message'].progress_map(translate_emoji)
misspeling = {

    "ñ": "não",

    "n": "não",

    "e-mail" : "email",

    "e-mails": "emails"

}



misspeling_embed = {

    "td": "tudo",

    "smp": "sempre",

    "masss": "mas",

    "ecxelente": "excelente",

    "cx": "caixa",

    "ñ": "não",

    "n": "não",

    "e-mail" : "email",

    "e-mails": "emails",

    "qdo": "quando",

    "q": "que",

    "pq": "porque",

    "msm": "mesmo",

    "geito": "jeito",

    "vcs": "vocês",

    "vc": "você",

    "obg": "obrigado",

    "cm": "centímetros",

    "entega" : "entrega",

    "und": "unidades",

    "adaquirido": "adquirido",

    "pedente": "pendente",

    "eh": "é",

    "mto": "muito",

    "p": "para",

    "d": "de",

    "p/": "para",

    "nf": "nota fiscal",

    "c": "com",

    "c/": "com",

    "hj": "hoje",

    "plazo": "prazo",

    "corsspindente": "correspondente",

    "gustei": "gostei",

    "oque": "o que",

    "r$": "reais",

    "on-line": "online",

    "talves": "talvez",

    "tb": "também",

    ":)": "legal",

    ":))": "legal",

}
# Define preprocessing function

import string

import unidecode

import unicodedata

import re





nlp = spacy.load('pt', parser=False, entity=False)



def preprocessing(x,embed):

    

    x = x.lower()

    

    # misspelling

    miss_res = []

    misspel = {}

    if embed:

        misspel = misspeling_embed

    else:

        misspel = misspeling

    for item in x.split():

        # iterate by keys

        if item in misspel.keys():

            # look up and replace

            miss_res.append(misspel[item])

        else:

            miss_res.append(item)

    

    x = ' '.join(miss_res)

    

    # Remove links

    x = re.sub(r'http\S+', ' ', x)

    

    # Remove punct

    x = re.sub(r'[^\w\s]',' punct ', x )

    

    # Remove extra chars

    x = re.sub(r'([a-z])\1+', r'\1', x)

    

    # Remove accents

    x = unicodedata.normalize('NFD', x)

    x = x.encode('ascii', 'ignore')

    x = x.decode("utf-8")

    

    x = ' '.join(x.split())

    

    return x
#Generate embedding

from gensim.models.fasttext import FastText as FT_gensim

from gensim.test.utils import datapath, common_texts, get_tmpfile

from gensim.models import Word2Vec



#Pre process text

train_new = train_df

test_new = test_df

train_new['review_comment_message'] = train_df['review_comment_message'].progress_map(lambda x :preprocessing(x, True))

test_new['review_comment_message'] = test_df['review_comment_message'].progress_map(lambda x : preprocessing(x, True))

sentences_emb = (train_new['review_comment_message'].tolist() + test_new["review_comment_message"].tolist())
#Generate corpus

with open('word_embeddings/corpus', 'w', encoding='utf-8') as f:

        for sentence in tqdm(sentences_emb):

            f.write(sentence + '\n')
#Generate Model

def generate_model():

    model_gensim = FT_gensim(size=300)



    # build the vocabulary

    model_gensim.build_vocab(corpus_file='word_embeddings/corpus')



    # train the model

    model_gensim.train(

        corpus_file='word_embeddings/corpus', epochs=model_gensim.epochs,

        total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words

    )



    model_gensim.save('word_embeddings/fasttext.vec')

    

#generate_model()
# Preprocessing

train_df['review_comment_message'] = train_df['review_comment_message'].progress_map(lambda x :preprocessing(x, False))

train_df.head()
test_df['review_comment_message'] = test_df['review_comment_message'].progress_map(lambda x :preprocessing(x, False))
# Define values

seq_size     = 50

max_tokens   = 15274

embed_dim    = 300
#Generate X e Y

## ** Do not use text_fit for texts to sequecen, because X and Y are different

tokenizer = Tokenizer(num_words=max_tokens, split=' ')



# Join text from train and test for fit

text_fit = list(train_df['review_comment_message'].values) + list(test_df['review_comment_message'].values)



# Generate text with only train text, for sequences

text = train_df['review_comment_message'].values



tokenizer.fit_on_texts(text_fit)



X = tokenizer.texts_to_sequences(text)  



X = pad_sequences(X, maxlen=seq_size)



Y = train_df['review_score'].values
# Generate embeddings from glove, fast and word2vec

# ** Due limitation of disk on Kaggle, was not possible download all embeddings



def embedding(tok,embedding_file,max_features,embed_size):

    print("Gerando Embedding")





    embeddings_index = {}

    with open(embedding_file,encoding='utf8') as f:

        for line in f:

            values = line.rstrip().rsplit(' ')

            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs



    word_index = tok.word_index

    #prepare embedding matrix

    num_words = min(max_features, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector



    return embedding_matrix



#embedding_f = embedding(tokenizer, 'word_embeddings/skip_s300.txt', max_tokens,embed_dim)

embedding_g = embedding(tokenizer, 'word_embeddings/g.txt', max_tokens,embed_dim)

#embedding_w = embedding(tokenizer, 'word_embeddings/cbow_s300.txt', max_tokens,embed_dim)

#embedding_matrix = np.concatenate((embedding_f, embedding_g,embedding_w), axis=1)
# Make load of embedding gerated previosly



from gensim.models.fasttext import FastText as FT_gensim

def generated_embedding(tok,max_features,embed_size):

    print("Fasttext Embedding")



    loaded_model = FT_gensim.load('/kaggle/input/ecembedding/fasttext.vec')



    word_index = tok.word_index

    # prepare embedding matrix

    num_words = min(max_features, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, embed_size))

    unknown_vector = np.random.normal(size=embed_size)

    for key, i in word_index.items():

        if i >= max_features:

            continue

        word = key

        embedding_vector = loaded_model[word]

        embedding_matrix[i] = embedding_vector



    return embedding_matrix



#embedding_o = generated_embedding(tokenizer, max_tokens,embed_dim)
#embedding_matrix = np.concatenate((embedding_g, embedding_o), axis=1)

embedding_matrix = embedding_g
# Multiply o embed_dim by quantity of embeddings

from keras.models import Model

def get_model(embedding):

    sequence_input = Input(shape=(seq_size,))

    x = Embedding(max_tokens, embed_dim, weights=[embedding], trainable=True, input_length = seq_size)(sequence_input)

    x = SpatialDropout1D(0.3)(x)

    

    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

    x2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)

    x1_conv = Conv1D(256, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x1)

    

    max_pool1 = GlobalMaxPooling1D()(x1)

    max_pool2 = GlobalMaxPooling1D()(x2)

    max_pool3 = GlobalMaxPooling1D()(x1_conv)

    concat_1 = concatenate([max_pool1, max_pool2,max_pool3])

    

    dense = Dense(256, activation='relu')(concat_1)

    

    preds = Dense(1, activation='relu')(dense)

    

    model = Model(sequence_input, preds)

    model.compile(loss =tf.keras.losses.Huber(), optimizer='rmsprop', metrics=['accuracy'])

    

    #model.summary()

    return model
from sklearn.model_selection import ShuffleSplit

from imblearn.over_sampling import SMOTE,RandomOverSampler

import itertools



skf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=133)



for n_fold, (train_indices, val_indices) in enumerate(skf.split(X,Y)):

    

    base_model = get_model(embedding_matrix)

    

    X_train = X[train_indices]

    Y_train = Y[train_indices]

    X_valid = X[val_indices]

    Y_valid = Y[val_indices]

    

    filepath = 'best_model_' + str(n_fold) + '.h5'



    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



    class_weights = class_weight.compute_class_weight('balanced', train_df["review_score"].unique(), Y_train)



    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)



    reduce_lr = ReduceLROnPlateau(

                    monitor  = 'val_loss',

                    factor   = 0.3,

                    patience = 1,

                    verbose  = 1,

                    mode     = 'auto',

                    epsilon  = 0.0001,

                    cooldown = 0,

                    min_lr   = 0

                )



    callbacks_list = [reduce_lr, early, checkpoint]



    hist = base_model.fit(X_train, Y_train, 

                  validation_data =(X_valid, Y_valid),

                  class_weight=class_weights,

                  batch_size=8, nb_epoch = 30,  verbose = 1, callbacks=callbacks_list)
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])
# Evaluate one fold



def eval_fold():

    

    base_model = get_model(embedding_matrix)

    base_model.load_weights('best_model_4.h5')

    

    final_predict = base_model.predict(X_valid)

    

    return final_predict
val_predict = eval_fold()

val_predict =  val_predict[:,0]

val_predict
Y_valid
rms = sqrt(mean_squared_error(Y_valid, val_predict))

rms
#Gerar melhores valores para arredondar

# Generate best values for round

def arrendondar(n_min,n_max, score):

    n = int(str(score).split('.')[1][:2])

    if n < n_min:

        return math.floor(score)

    elif n >= n_max:

        return math.ceil(score)

    else:

        return score
old_rms = 1

best_result = ''

val_n_min = 0

val_n_max = 0

for n_min in np.arange(0, 50, 1):

    for n_max in np.arange(50, 100, 1):

        new_val = np.array([arrendondar(n_min,n_max,x) for x in val_predict])

        rms = sqrt(mean_squared_error(Y_valid, new_val))

        if rms < old_rms:

            val_n_min = n_min

            val_n_max = n_max

            best_result = 'N_min {}, N_max {}, rms {}'.format(n_min,n_max,rms)

            old_rms = rms

print(best_result)
def predict(text):

    new_text = tokenizer.texts_to_sequences(text)

    new_text = pad_sequences(new_text, maxlen=seq_size)

    pred     = base_model.predict(new_text)

    

    return pred
def predict_fold(text):

    new_text = tokenizer.texts_to_sequences(text)

    new_text = pad_sequences(new_text, maxlen=seq_size)

    predict_folds=[]

    for n in range(5):

        base_model = get_model(embedding_matrix)

        base_model.load_weights('best_model_' + str(n) + '.h5')

        predict_folds.append(base_model.predict(new_text))

    

    final_predict = np.mean(predict_folds, axis=0)

    

    return final_predict
# Predict for all folds

pred     = predict_fold(test_df.review_comment_message)

pred     = pred[:,0]

pred[:5] 
test_df.head()
test_df['review_score'] = pred

test_df.head()
test_df['review_score'] = test_df['review_score'].apply(lambda x: arrendondar(val_n_min,val_n_max,x))

test_df.head()
test_df[['review_id', 'review_score']].to_csv('submission.csv', index=False)