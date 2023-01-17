# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from keras.preprocessing import text,sequence

from gensim.models import KeyedVectors

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')

### creating new features 

## check for capitals and exclamation makrs and 

train['capitals'] = train['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))

train['exclamation_points'] = train['comment_text'].apply(lambda x: len(regex.findall(x)))

train['total_length'] = train['comment_text'].apply(len)
features_added=('capitals','exclamation_points','total_length')

features_existing= ('target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 

                     'threat','funny', 'wow', 'sad', 'likes', 'disagree', 'sexual_explicit',

                     'identity_annotator_count', 'toxicity_annotator_count')



        

rows = [{c:train[f].corr(train[c]) for c in features_existing} for f in features_added]

train_correlations = pd.DataFrame(rows, index=features_added)

sns.set()

sns.heatmap(train_correlations)



## since there are not highly coreelated we can use these as standalone features

## for datat preprocessing with lstm netwroks we will use the target variable

max_len = 200## each word toke needs to be aroud 200 characters

filter_char = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

x_train = train['comment_text']

x_test=test['comment_text']

tokenizer = text.Tokenizer(filters=filter_char)

tokenizer.fit_on_texts(list(x_train)+ list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)

x_test= tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)

x_test = sequence.pad_sequences(x_test, maxlen=max_len)
IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]



TARGET_COLUMN = 'target'



for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train[column]= np.where(train[column] >= 0.5, True, False)
embedding_files = ['../input/gensim-embeddings-dataset/crawl-300d-2M.gensim','../input/gensim-embeddings-dataset/glove.840B.300d.gensim' ]
def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(512, activation='relu')(hidden)])

    hidden = add([hidden, Dense(512, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model



def build_matrix(word_index, path):

    embedding_index = KeyedVectors.load(path, mmap='r')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        for candidate in [word, word.lower()]:

            if candidate in embedding_index:

                embedding_matrix[i]=embedding_index[candidate]

                break

                

    return embedding_matrix



tokenizer.word_index

embedding_matrix=np.concatenate([build_matrix(tokenizer.word_index,path) for path in embedding_files],axis=-1)



BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TARGET_COLUMN = 'target'

y_train = train[TARGET_COLUMN].values

y_aux_train = train[AUX_COLUMNS].values



model = build_model(embedding_matrix, y_aux_train.shape[-1])

model_his = model.fit(x_train,[y_train, y_aux_train],batch_size=BATCH_SIZE,epochs=4,verbose=2)
predictions = model.predict(x_test, batch_size=2048)[0].flatten()

predictions
submission = pd.DataFrame.from_dict({

    'id': test.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)