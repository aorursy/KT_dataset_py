# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import gc



from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding

from keras.layers import Input

from keras.layers import Conv1D

from keras.layers import MaxPooling1D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.models import Model

from keras.models import load_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")

sample_submission.head(5)
train_file_path='/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl'

test_file_path='/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl'



#written by ragnar

def read_data(path, sample = True, chunksize = 30000):

    if sample == True:

        df = []

        with open(path, 'r') as reader:

            for i in range(chunksize):

                df.append(json.loads(reader.readline()))

        df = pd.DataFrame(df)

        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

    else:

        df = pd.read_json(path, orient = 'records', lines = True)

        print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

        gc.collect()

    return df



train = read_data(train_file_path, sample = True)

test = read_data(test_file_path, sample = False)

train.head()
def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        # written by MJ Bahmani

        for col in df.columns:

            dtype = str(train[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)

    

check_missing_data(train)

check_missing_data(test)
from nltk.tokenize import sent_tokenize, word_tokenize

sample_text=train.document_text[0]

phrases = sent_tokenize(sample_text)

words = word_tokenize(sample_text)

print(phrases)
print(words)
type(words)
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords



stopWords = set(stopwords.words('english'))

words = word_tokenize(sample_text)

wordsFiltered = []

 

for w in words:

    if w not in stopWords:

        wordsFiltered.append(w)

 

print(wordsFiltered)
from wordcloud import WordCloud as wc

from nltk.corpus import stopwords 

import matplotlib.pyplot as plt

from nltk.corpus import stopwords



eng_stopwords = set(stopwords.words("english"))



def generate_wordcloud(text): 

    wordcloud = wc(relative_scaling = 1.0,stopwords = eng_stopwords).generate(text)

    fig,ax = plt.subplots(1,1,figsize=(10,10))

    ax.imshow(wordcloud, interpolation='bilinear')

    ax.axis("off")

    ax.margins(x=0, y=0)

    plt.show()

    

generate_wordcloud(train.document_text[0])
MAX_NUM_WORDS = 10000

TEXT_COLUMN = 'question_text'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(train[TEXT_COLUMN])



# All comments must be truncated or padded to be the same length.

MAX_SEQUENCE_LENGTH = 250

def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
EMBEDDINGS_PATH = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'

EMBEDDINGS_DIMENSION = 100

DROPOUT_RATE = 0.3

LEARNING_RATE = 0.00005

NUM_EPOCHS = 10

BATCH_SIZE = 128



def train_model(train_df, test, tokenizer):

    # Prepare data

    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)

   # train_labels = to_categorical(train_df[TOXICITY_COLUMN])

  #  validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)

   # validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])



    # Load embeddings

    print('loading embeddings')

    embeddings_index = {}

    with open(EMBEDDINGS_PATH) as f:

        for line in f:

            values = line.split()

            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs



    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,

                                 EMBEDDINGS_DIMENSION))

    num_words_in_embedding = 0

    for word, i in tokenizer.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            num_words_in_embedding += 1

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector



    # Create model layers.

    def get_convolutional_neural_net_layers():

        """Returns (input_layer, output_layer)"""

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        embedding_layer = Embedding(len(tokenizer.word_index) + 1,

                                    EMBEDDINGS_DIMENSION,

                                    weights=[embedding_matrix],

                                    input_length=MAX_SEQUENCE_LENGTH,

                                    trainable=False)

        x = embedding_layer(sequence_input)

        x = Conv1D(128, 2, activation='relu', padding='same')(x)

        x = MaxPooling1D(5, padding='same')(x)

        x = Conv1D(128, 3, activation='relu', padding='same')(x)

        x = MaxPooling1D(5, padding='same')(x)

        x = Conv1D(128, 4, activation='relu', padding='same')(x)

        x = MaxPooling1D(40, padding='same')(x)

        x = Flatten()(x)

        x = Dropout(DROPOUT_RATE)(x)

        x = Dense(128, activation='relu')(x)

        preds = Dense(2, activation='softmax')(x)

        return sequence_input, preds



    # Compile model.

    print('compiling model')

    input_layer, output_layer = get_convolutional_neural_net_layers()

    model = Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy',

                  optimizer=RMSprop(lr=LEARNING_RATE),

                  metrics=['acc'])



    # Train model.

    print('training model')

    model.fit(train_text,

              train_labels,

              batch_size=BATCH_SIZE,

              epochs=NUM_EPOCHS,

              validation_data=(validate_text, validate_labels),

              verbose=2)



    return model



model = train_model(train, test, tokenizer)