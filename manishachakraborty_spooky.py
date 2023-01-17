

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import string

import numpy as np 

import pandas as pd 

import os

from tqdm import tqdm

import re

import statistics

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from keras.optimizers import SGD



from keras.preprocessing import sequence, text

from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.layers.core import Dense,Dropout, Activation

from keras.layers.embeddings import Embedding

from keras.layers import  Bidirectional, SpatialDropout1D, GlobalMaxPooling1D

from keras.utils import np_utils

from keras.callbacks import EarlyStopping

from keras.utils.vis_utils import plot_model

from keras.utils import to_categorical





from nltk import word_tokenize

from nltk.corpus import stopwords

stopwords = stopwords.words('english')





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/spooky-author-identification/train.zip')

test = pd.read_csv('/kaggle/input/spooky-author-identification/test.zip')

sample = pd.read_csv('/kaggle/input/spooky-author-identification/sample_submission.zip')
train.head()

train.isnull().sum()

sample.head()

test.head()
def label_encoding(text):

    lbl_enc = LabelEncoder()

    integer_encoding = lbl_enc.fit_transform(text)

    return integer_encoding



train['author_integer_encode'] = label_encoding (train['author'])
y_train = to_categorical(train['author_integer_encode'],3)

y_train
y = pd.DataFrame(y_train,columns=['EAP','HPL','MWS'])

y.head()
train.head()

def replace_typical_misspell(text):

        miss_spell = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",

              "didn't": "did not", "doesn't": "does not", "don't": "do not",

              "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                 "he'd": "he would", "he'll": "he will", "he's": "he is",

                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",

                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",

                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",

                 "she'd": "she would", "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",

                 "they'd": "they would", "they'll": "they will", "they're": "they are",

                 "they've": "they have", "we'd": "we would", "we're": "we are",

                 "weren't": "were not", "we've": "we have", "what'll": "what will",

                 "what're": "what are", "what's": "what is", "what've": "what have",

                 "where's": "where is", "who'd": "who would", "who'll": "who will",

                 "who're": "who are", "who's": "who is", "who've": "who have",

                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",

                 "you'll": "you will", "you're": "you are", "you've": "you have",

                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"

             }

        misspell_re = re.compile('(%s)' %'|'.join(miss_spell.keys()))

        def replace(match):

            return miss_spell[match.group(0)]

        return misspell_re.sub(replace,text)

   

    
def remove_punctuations(text):

    extra_chars = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',

          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',

          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',

          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',

          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',

          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',

          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

    text =''.join(x for x in str(text) if x not in list(string.punctuation)+extra_chars)

    return text

    
def remove_digits(text):

    return re.sub(r'\d+',' ',text)


train['text']=train['text'].apply(replace_typical_misspell)

train['text'] = train['text'].apply(remove_punctuations)

train['text']= train['text'].apply(remove_digits)
test['text']=test['text'].apply(replace_typical_misspell)

test['text'] = test['text'].apply(remove_punctuations)

test['text']= test['text'].apply(remove_digits)
train.text.values
def info(df):

    df['line_num'] = train['text'].apply(lambda x : len(x.split()))

    max_len = max(df['line_num'])

    min_len = min(df['line_num'])

    median= statistics.median(df['line_num'])

    count = Counter(df['line_num'])

    return max_len, min_len, median,count

    

max_len, min_len, median,count = info(train)

print('Train set: \n maximum number of words: {0}\n minimum number of words: {1} \n Median of the number of words: {2} \n Varying lengths of the texts and their frequencies : \n {3} \n '.format(max_len,min_len,median,count))

print('number of samples: {}'.format(train.shape[0]))
def loading_embedding(path):

    def get_coeffs(word, *a):

        return word,np.asarray(a,dtype = np.float32)

    embeddings =  dict(get_coeffs(*o.rstrip().split(" ")) for o in open(path) if len(o) >300  ) 

    return embeddings 



fast_vecs = loading_embedding('/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec')
maxlen = 123

max_features = 15000

embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train['text'])+list(test['text']))



x_train = tokenizer.texts_to_sequences(train['text'])

x_train = sequence.pad_sequences(x_train,maxlen=maxlen, padding = 'post', truncating= 'pre')



y_test = tokenizer.texts_to_sequences(test['text'])

y_test = sequence.pad_sequences(y_test,maxlen=maxlen, padding= 'post',truncating='pre')

    
def embedding_matrix(fast_vecs, word_index,max_features):

    fast_vals = np.stack(fast_vecs.values())

    embed_size = fast_vals.shape[1]

    matrix = np.random.normal(size =(max_features,embed_size))

    for word,i in word_index.items():

        if i < max_features:

            embedding_vector = fast_vecs.get(word)

            if embedding_vector is not None:

                matrix[i]=embedding_vector

    return matrix

matrix = embedding_matrix(fast_vecs,tokenizer.word_index,max_features)
matrix.shape
def Model(max_features, embedding_size, embedding_matrix):

    model = Sequential()

    model.add(Embedding(max_features,embedding_size,weights = [embedding_matrix], input_length=123, trainable = False))

    model.add(Bidirectional(LSTM(256, return_sequences = True,unit_forget_bias= True)))

    model.add(SpatialDropout1D(0.2))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(128,activation='sigmoid' ))

    model.add(Dropout(0.5))

    model.add(Dense(3,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model
model = Model(max_features,embed_size,matrix)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
es = EarlyStopping(monitor='val_loss', mode='min',verbose=1)

history = model.fit(x_train,y,batch_size=64,epochs=25,verbose=1,validation_split=0.2,callbacks=[es])
submit = model.predict_proba(y_test)
sample.head()

test.head()


ids = test['id']

predict = pd.DataFrame(submit, columns=['EAP','HPL','MWS'])

submission = pd.concat([ids, predict] ,axis = 1)

submission.to_csv('submission.csv',index=False)