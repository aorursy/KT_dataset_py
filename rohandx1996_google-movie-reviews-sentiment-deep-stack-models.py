# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import pandas as pd

from nltk.corpus import stopwords

import re

import os

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/word2vec-nlp-tutorial/"))

print(os.listdir("../input/movie-review/"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

df_train.head()
df_train1=pd.read_csv("../input/movie-review/imdb_master.csv",encoding="latin-1")

df_train1.head()
df_train1=df_train1.drop(["type",'file'],axis=1)
df_train1.rename(columns={'label':'sentiment',

                          'Unnamed: 0':'id',

                          'review':'review'}, 

                 inplace=True)
df_train1 = df_train1[df_train1.sentiment != 'unsup']
maping = {'pos': 1, 'neg': 0}

df_train1['sentiment'] = df_train1['sentiment'].map(maping)
new_train=pd.concat([df_train,df_train1])
df_test=pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv",header=0, delimiter="\t", quoting=3)

df_test.head()
new_train.head()
from bs4 import BeautifulSoup

def review_to_words( raw_review ):

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review, 'lxml').get_text() 

    

    # 2. Remove non-letters with regex

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                           

    

    # 4. Create set of stopwords

    stops = set(stopwords.words("english"))                  

    

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))   



new_train['review']=new_train['review'].apply(review_to_words)

df_test["review"]=df_test["review"].apply(review_to_words)
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(new_train["review"])
# checking nullity in the data of train and test

new_train.isnull().sum(),df_test.isnull().sum()

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
list_classes = ["sentiment"]

y = new_train[list_classes].values

list_sentences_train = new_train["review"]

list_sentences_test = df_test["review"]
max_features = 6000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])

plt.xlabel("Distribution of comment")

plt.ylabel("no of comments")

plt.title("no of comments vs no of words distribution ")

plt.show()
maxlen = 370

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
inp = Input(shape=(maxlen, ))

embed_size = 128

x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
batch_size = 32

epochs = 2

model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
prediction = model.predict(X_te)

y_pred = (prediction > 0.5)
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)

y_test = df_test["sentiment"]
from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))

print('Confusion matrix:')

confusion_matrix(y_pred, y_test)
# ouput submission file 

df_test = df_test[['id','sentiment']]
df_test.to_csv("submission.csv",index=False)