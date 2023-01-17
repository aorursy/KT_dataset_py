import numpy as np

import re

import string

import pandas as pd

from nltk.corpus import stopwords





def DataCleaning(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = text.lower().split()

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops]

    text = " ".join(text)

    return (text)





def Clean(text):

    text = DataCleaning(text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text





def clean_data():

    path = '../input/fakenewsdata/train.csv'

    vector_dimension=300



    data = pd.read_csv(path)



    missing_rows = []

    for i in range(len(data)):

        if data.loc[i, 'text'] != data.loc[i, 'text']:

            missing_rows.append(i)

    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)



    for i in range(len(data)):

        data.loc[i, 'text'] = Clean(data.loc[i,'text'])



    data = data.sample(frac=1).reset_index(drop=True)



    x = data.loc[:,'text'].values

    y = data.loc[:,'label'].values



    return x,y



xtrain,ytrain =clean_data()
path = '../input/fakenewsdata/train.csv'

data = pd.read_csv(path)

data1=pd.DataFrame()

data1['word_count'] = data['text'].apply(lambda x: len(str(x).split(" ")))

data1['Text']=data['text']

data1[['Text','word_count']].head()
import seaborn as sns

wh1 = data[['title','author',

            'text','label']] #Subsetting the data

cor = wh1.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS=12000



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(xtrain)

sequences = tokenizer.texts_to_sequences(xtrain)



word_index = tokenizer.word_index





x_traindata = pad_sequences(sequences, maxlen=200)



y_labels = to_categorical(np.asarray(ytrain))



vocab_size = MAX_NB_WORDS



wordindex=tokenizer.word_index



def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    max_features=MAX_NB_WORDS

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix



embedding_matrix=load_fasttext(wordindex)
#data = pd.DataFrame(x_traindata, columns=['x', 'y'])





plt.hist(x_traindata, normed=True, alpha=0.5)

plt.show()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Embedding

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

model = Sequential()

e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=200, trainable=False)

model.add(e)

model.add(Dropout(0.2))

model.add(Conv1D(64, 5, activation='relu'))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(150))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


  



import matplotlib.pyplot as plt



history = model.fit(x_traindata, ytrain, epochs=2, verbose=1)



plt.plot(history.history['acc'])



plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.utils import plot_model

plot_model(model, to_file='model.png')
import pandas as pd

import nltk

from nltk.probability import FreqDist

import seaborn as sb

from matplotlib import pyplot as plt

path = '../input/fakenewsdata/train.csv'



data = pd.read_csv(path)

text=data['text']

labels=data['label']

from matplotlib import pyplot as plt

freqdist = nltk.FreqDist(text)

plt.figure(figsize=(16,5))

freqdist.plot(50)

inputnews="For individuals like Margaret Thome Bekema, finishing high school was a dream she didnt think would ever come true. Instead of graduating from Grand Rapids Catholic "

testd=inputnews.lower()



tokenizer2 = Tokenizer(nb_words=MAX_NB_WORDS)

tokenizer2.fit_on_texts(xtrain)

sequences = tokenizer2.texts_to_sequences(testd)

    

newsequences=np.asarray(sequences)

sequences=np.reshape(newsequences,(len(newsequences),))

n =[]

for i in sequences:

    n= n+i

    

newsequences=np.asarray(n)    

x_testdata = pad_sequences([newsequences], maxlen=200)

prediction = model.predict_classes(x_testdata)

prediction


    

