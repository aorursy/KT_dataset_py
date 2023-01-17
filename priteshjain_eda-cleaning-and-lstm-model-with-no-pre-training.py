import pandas as pd

import numpy as np

import os

import re



import tensorflow

from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense, concatenate

from tensorflow.keras.models import Model

from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import EarlyStopping
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train.shape, test.shape
train.head()
train.keyword.unique()[:10]
train.keyword.nunique()
test.keyword.unique()[:10]
test.keyword.nunique()
train.groupby('keyword')['target'].mean().head()
train.location.unique()[:10]
test.location.unique()[:10]
train.location.nunique()
test.location.nunique()
train.text[0:5]
test.text[0:5]
def clean(my_string):

    if not pd.isna(my_string):

        return ' '.join(str.lower(re.sub(r'[\W]', ' ', my_string)).split())

    else: return 'nan'
train['text'] = train['text'].map(clean)

train['keyword'] = train['keyword'].map(clean)

train['location'] = train['location'].map(clean)

test['text'] = test['text'].map(clean)

test['keyword'] = test['keyword'].map(clean)

test['location'] = test['location'].map(clean)
train['text'].head()
train['location'].unique()
train['text_len'] = train['text'].apply(lambda x: len(x.split()))

test['text_len'] = test['text'].apply(lambda x: len(x.split()))
train['text_len'].max(), test['text_len'].max()
all_words = pd.Series((' '.join(train['text']) + ' '.join(test['text'])).split()).value_counts().reset_index()
all_words.columns = ['word', 'count']

all_words.head()
all_words.shape
all_words[all_words['count']>2].shape
words = all_words[all_words['count']>1]['word'].values

len(words)
word_dict = {}

for i in range(len(words)):

    word_dict[words[i]] = i+1



def get_word_index(word):

    if word in word_dict.keys():

        return word_dict[word]

    else: return len(word_dict)
keywords = pd.concat([train['keyword'], test['keyword']])

keywords = set(keywords)

keyword_dict = {k:v+1 for v,k in enumerate(keywords)}



def get_keyword_index(keyword):

    if keyword in keyword_dict.keys():

        return keyword_dict[keyword]

    else: return len(keyword_dict)
class DataGenerator(Sequence):

    def __init__(self, input_df, batch_size=64):

        self.batch_size = batch_size

        self.input_df = input_df

        self.ids = input_df.index.unique()



    def __len__(self):

        return int(np.floor(len(self.input_df) / self.batch_size))



    def __getitem__(self, index):

        sample_ids = np.random.choice(self.ids, self.batch_size)

        text_input, keyword_input, target = self.__data_generation(sample_ids)

        return [text_input, keyword_input],[np.reshape(target, target.shape + (1,))]



    def __data_generation(self, ids):

        max_length = 34

        text_input = []

        target = []

        keyword_input = []

        for id in ids:

            text = self.input_df['text'][id].split()

            text = [get_word_index(word) for word in text]

            keyword = [get_keyword_index(self.input_df['keyword'][id])]*len(text)

            extend_length = max_length - len(text)

            if extend_length > 0:

                text = np.append(text, [0]*extend_length)

                keyword = np.append(keyword, [0]*extend_length)

            text_input.append(text)

            keyword_input.append(keyword)

            target.append(self.input_df['target'][id])

        return text_input, keyword_input, np.array(target, dtype='float64')

                
tensorflow.keras.backend.clear_session()

tensorflow.compat.v1.disable_v2_behavior()





text_input = Input((None, ), dtype='float64', name='text_input')

text = Embedding(len(word_dict)+1, 30, mask_zero=True, dtype='float64', name='text')(text_input)



keyword_input = Input((None,), dtype='float64', name='keyword_input')

keyword = Embedding(len(keyword_dict)+1, 20, mask_zero=True, dtype='float64')(keyword_input)



inputs = concatenate([keyword, text], dtype='float64', name='inputs')

lstm = LSTM(50,dtype='float64', name='lstm')(inputs)

dropout = Dropout(0.3,dtype='float64', name='dropout')(lstm)

output = Dense(1, activation='sigmoid', dtype='float64', name='output')(dropout)

model = Model(inputs=[text_input, keyword_input], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='rmsprop')
training_id = pd.Series(train.index.values)

train_ids = training_id.sample(frac=0.9)

val_ids = training_id[~training_id.isin(train_ids)]



train_data = train[train.index.isin(train_ids)]

val_data = train[train.index.isin(val_ids)]



trainingGenerator = DataGenerator(train_data, batch_size=64)

validationGenerator = DataGenerator(val_data, batch_size=64)
train_data.head()
model.fit_generator(trainingGenerator, 

                    validation_data=validationGenerator, 

                    epochs=10,

                    use_multiprocessing=False,

                    callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1)])
class PredictDataGenerator(Sequence):

    def __init__(self, input_df, batch_size=64):

        self.batch_size = batch_size

        self.input_df = input_df

        self.ids = input_df.index.unique()



    def __len__(self):

        return int(np.floor(len(self.input_df) / self.batch_size))



    def __getitem__(self, index):

        sample_ids = self.ids

        text_input, keyword_input = self.__data_generation(sample_ids)

        return [text_input, keyword_input]



    def __data_generation(self, ids):

        max_length = 34

        text_input = []

        keyword_input = []

        for id in ids:

            text = self.input_df['text'][id].split()

            text = [get_word_index(word) for word in text]

            keyword = [get_keyword_index(self.input_df['keyword'][id])]*len(text)

            extend_length = max_length - len(text)

            if extend_length > 0:

                text = np.append(text, [0]*extend_length)

                keyword = np.append(keyword, [0]*extend_length)

            text_input.append(text)

            keyword_input.append(keyword)

        return text_input, keyword_input

                

                

testGenerator = PredictDataGenerator(test, batch_size=len(test))

preds = model.predict_generator(testGenerator)
predictions = []

for i in preds:

    val = 1 if i>0.5 else 0

    predictions.append(val)

submit = pd.DataFrame()

submit['id'] = test['id']

submit['target'] = predictions

submit.to_csv('submission.csv', index=False)