import numpy as np

import re

import string

import pandas as pd

from nltk.corpus import stopwords





def DataCleaning(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/+-=]"," ", text)

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

    x = []

    y = []



    data = pd.read_csv(path)

    for key,row in data.iterrows():

        

        x.append(DataCleaning(str(row['title'])))

        y.append(row['label'])

        



    return x,y



xtrain,ytrain =clean_data()
path = '../input/fakenewsdata/train.csv'

    



data = pd.read_csv(path)

data.head()
xtrainlist = []

for line in xtrain:

    xtrainlist.append([line])
xtrainlist
xtrain = np.asarray(xtrain)
np.save('xtrain.npy',xtrain)
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
xtraintokens = []

for line in xtrain:

    line = str(line)

    xtraintokens.append(line.split(' '))
totaltrain = []

for i in xtrain:

    for j in i.split(' '):

        totaltrain.append(j)
from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS=12000



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)



tokenizer.fit_on_texts(totaltrain)

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
x_traindata
import matplotlib.pyplot as plt
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

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
xtrain[3]


x_traindata[5]

ytrain[3]


import matplotlib.pyplot as plt



history = model.fit(x_traindata, ytrain, epochs=2, verbose=1,validation_split=0.1)



model.save()
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

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 

attention = Flatten()(attention)

attention = Activation('softmax')(attention)

attention = RepeatVector(units)(attention)

attention = Permute([2, 1])(attention)



# apply the attention

sent_representation = merge([activations, attention], mode='mul')

sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
ytrain[37]

inputnews=xtrain[37]



sequences = tokenizer.texts_to_sequences(inputnews.split(' '))

newsequences=np.asarray(sequences)

sequences=np.reshape(newsequences,(len(newsequences),))

n =[]

for i in sequences:

    n= n+i  

newsequences=np.asarray(n)    

x_testdata = pad_sequences([newsequences], maxlen=200)

prediction = model.predict_classes(x_testdata)

percent =  model.predict(x_testdata)

percent