from datetime import datetime

startTime = datetime.now()



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

from tqdm import tqdm

from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate

import os

import pandas as pd

import numpy as np

import re

import gc



import json



import numpy as np

from tqdm import tqdm

from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate



import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt



import numpy as np

from tqdm import tqdm

from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate

import os

import pandas as pd

import numpy as np

import re

import gc

import numpy as np

from tqdm import tqdm

from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate

import os

import pandas as pd

import numpy as np

import re

import gc



import json



import numpy as np

from tqdm import tqdm

from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate



import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt



from keras.layers import Input, Embedding, LSTM, Dense , concatenate, Bidirectional,CuDNNLSTM

from keras.models import Model

import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from gensim.models import KeyedVectors
def clean_text(x):



    x = str(x)

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^`{|}~' + '“”’':

        x = x.replace(punct, f' {punct} ')

   

    for punct in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—_':

        x = x.replace(punct, f' {punct} ')

 

    return x



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {"usepackage" : "use package",

                'instrumentsettingsid':'instrumental settings id',

                'RippleShaderProgram' : 'ripple shader program',

                'ShaderProgramConstants':'shader program constants',

                'storedElements':'stored elements',

                'stackSize' : 'stack size',

                '_':' '



                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)
train_df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
train_df["question_body"] = train_df["question_body"].apply(lambda x: clean_text(x))

train_df["question_body"] = train_df["question_body"].apply(lambda x: replace_typical_misspell(x))



train_df["question_title"] = train_df["question_title"].apply(lambda x: clean_text(x))

train_df["question_title"] = train_df["question_title"].apply(lambda x: replace_typical_misspell(x))



train_df["category"] = train_df["category"].apply(lambda x: clean_text(x))

train_df["category"] = train_df["category"].apply(lambda x: replace_typical_misspell(x))



train_df["answer"] = train_df["answer"].apply(lambda x: clean_text(x))

train_df["answer"] = train_df["answer"].apply(lambda x: replace_typical_misspell(x))



test_df["question_body"] = test_df["question_body"].apply(lambda x: clean_text(x))

test_df["question_body"] = test_df["question_body"].apply(lambda x: replace_typical_misspell(x))



test_df["question_title"] = test_df["question_title"].apply(lambda x: clean_text(x))

test_df["question_title"] = test_df["question_title"].apply(lambda x: replace_typical_misspell(x))



test_df["category"] = test_df["category"].apply(lambda x: clean_text(x))

test_df["category"] = test_df["category"].apply(lambda x: replace_typical_misspell(x))



test_df["answer"] = test_df["answer"].apply(lambda x: clean_text(x))

test_df["answer"] = test_df["answer"].apply(lambda x: replace_typical_misspell(x))
question_body = train_df['question_body']

answer = train_df['answer']

question_title = train_df["question_title"]

category = train_df["category"]



question_body_test = test_df['question_body']

answer_test = test_df['answer']

question_title_test = test_df["question_title"]

category_test = test_df["category"]
target = train_df[train_df.columns[-30:]]
all_text = pd.concat([train_df['question_body'],train_df['answer'],test_df['question_body'],test_df['answer'],train_df["question_title"],train_df["category"],test_df["question_title"],test_df["category"]])
tokenizer = Tokenizer(num_words=1000000, lower=False,filters='')



tokenizer.fit_on_texts(all_text)
question_body = tokenizer.texts_to_sequences(question_body)

answer = tokenizer.texts_to_sequences(answer)

question_title = tokenizer.texts_to_sequences(question_title)

category = tokenizer.texts_to_sequences(category)



question_body_test = tokenizer.texts_to_sequences(question_body_test)

answer_test = tokenizer.texts_to_sequences(answer_test)

question_title_test = tokenizer.texts_to_sequences(question_title_test)

category_test = tokenizer.texts_to_sequences(category_test)
lens = []

for i in question_body:

    lens.append(len(i))
vocab_size = len(tokenizer.word_index) + 1



maxlen = 245



question_body = pad_sequences(question_body, padding='post', maxlen=maxlen)

answer = pad_sequences(answer, padding='post', maxlen=maxlen)

question_title = pad_sequences(question_title, padding='post', maxlen=maxlen)

category = pad_sequences(category, padding='post', maxlen=maxlen)





question_body_test = pad_sequences(question_body_test, padding='post', maxlen=maxlen)

answer_test = pad_sequences(answer_test, padding='post', maxlen=maxlen)

question_title_test = pad_sequences(question_title_test, padding='post', maxlen=maxlen)

category_test = pad_sequences(category_test, padding='post', maxlen=maxlen)

from keras.layers import Input, Embedding, LSTM, Dense , concatenate, Bidirectional,CuDNNLSTM

from keras.models import Model

import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from gensim.models import KeyedVectors
def build_matrix(word_index, path):

    embedding_index = KeyedVectors.load(path, mmap='r')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        for candidate in [word, word.lower()]:

            if candidate in embedding_index:

                embedding_matrix[i] = embedding_index[candidate]

                break

    return embedding_matrix
EMBEDDING_FILES = [

    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',

    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'

]

NUM_MODELS = 3

BATCH_SIZE = 128

LSTM_UNITS = 64

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 10
embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
len(embedding_matrix)
inp1 = Input(shape=(None,))

inp2 = Input(shape=(None,))

inp3 = Input(shape=(None,))

inp4 = Input(shape=(None,))

words = concatenate([inp1,inp2,inp3,inp4])

x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)



hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

result = Dense(30, activation='sigmoid')(hidden)

model = Model(inputs=[inp1,inp2,inp3,inp4], outputs=[result])

model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae'])
print(datetime.now() - startTime)
model.fit(

           [question_body,question_title,category,answer], [target],

            batch_size=128,

            epochs=10,

            verbose=1,

        )
predictions = model.predict([question_body_test,question_title_test,category_test,answer_test])
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
sub = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")



for col_index, col in enumerate(target_cols):

    sub[col] = predictions[:, col_index]
sub
sub.to_csv("submission.csv", index = False)