import pandas as pd

import numpy as np

import gc

import re

from scipy.sparse import csr_matrix

from keras.models import Sequential

from keras.layers import CuDNNLSTM, Dense, Embedding, Dropout

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import roc_auc_score
def arrange_words(text):

    text = text.replace('!', '')

    text = text.replace('?', '')

    text = text.replace(',', '')

    text = text.replace('.', '')

    text = text.replace('“', '')

    text = text.replace('”', '')

    text = text.replace('‘', '')

    text = text.replace('’', '')

    text = text.replace('•', '')

    text = text.replace('・', '')

    text = text.replace('…', '')

    text = text.replace(':', '')

    text = text.replace(';', '')

    text = text.replace('(', '')

    text = text.replace(')', '')

    text = text.replace('{', '')

    text = text.replace('}', '')

    text = text.replace('[', '')

    text = text.replace(']', '')

    text = text.replace('<', '')

    text = text.replace('>', '')

    text = text.replace('\'', '')

    text = text.replace('\/', '')

    text = text.replace('"', '')

    text = text.replace('-', ' ')

    text = text.replace('_', ' ')

    text = text.replace('\n', ' ')

    text = text.replace('\r', ' ')

    text = text.replace('#', '')

    text = re.sub(r'[0-9]+', "0", text)

    text = ' ' + text + ' '

    return text
dtypes = {

        'id':                                             'category',

        'target':                                       'float16', 

        'comment_text':                           'category', 

        'severe_toxicity':                           'float16', 

        'obscene':                                    'float16', 

        'identity_attack':                           'float16', 

        'insult':                                         'float16', 

        'threat':                                        'float16', 

        'asian':                                         'float16', 

        'atheist':                                       'float16', 

        'bisexual':                                     'float16', 

        'black':                                         'float16', 

        'buddhist':                                    'float16', 

        'christian':                                    'float16', 

        'female':                                       'float16', 

        'heterosexual':                              'float16', 

        'hindu':                                         'float16', 

        'homosexual_gay_or_lesbian':        'float16', 

        'intellectual_or_learning_disability': 'float16', 

        'jewish':                                        'float16', 

        'latino':                                         'float16', 

        'male':                                          'float16', 

        'muslim':                                       'float16', 

        'other_disability':                           'float16', 

        'other_gender':                             'float16', 

        'other_race_or_ethnicity':              'float16', 

        'other_religion':                             'float16', 

        'other_sexual_orientation':             'float16', 

        'physical_disability':                       'float16', 

        'psychiatric_or_mental_illness':       'float16', 

        'transgender':                                'float16', 

        'white':                                          'float16', 

        'created_date':                              'category', 

        'publication_id':                             'category', 

        'parent_id':                                    'category', 

        'article_id':                                     'category', 

        'rating':                                         'category', 

        'funny':                                         'int8', 

        'wow':                                           'int8', 

        'sad':                                             'int8', 

        'likes':                                            'int8', 

        'disagree':                                     'int8', 

        'sexual_explicit':                             'float16', 

        'identity_annotator_count':             'int8', 

        'toxicity_annotator_count':             'int8', 

        }
train = pd.read_csv('../input/train.csv', dtype=dtypes)

test  = pd.read_csv('../input/test.csv',  dtype=dtypes)

train_ids = train.index

test_ids  = test.index

train_y = train['target'].apply(lambda x: 1 if x>=0.5 else 0)

train_X = train.drop('target', axis=1)

test_X = test

gc.collect()
train_X['comment_text_arranged'] = train_X['comment_text'].map(arrange_words)

test_X['comment_text_arranged'] = test_X['comment_text'].map(arrange_words)
tokenizer = Tokenizer(num_words=100000)

tokenizer.fit_on_texts(pd.concat([train_X['comment_text_arranged'], test_X['comment_text_arranged']]))

train_X_tokenized = tokenizer.texts_to_sequences(train_X['comment_text_arranged'])

test_X_tokenized = tokenizer.texts_to_sequences(test_X['comment_text_arranged'])
max_len = np.array([len(sentence) for sentence in train_X_tokenized + test_X_tokenized]).max()

train_X_padded = pad_sequences(train_X_tokenized, maxlen=max_len)

test_X_padded = pad_sequences(test_X_tokenized, maxlen=max_len)
test_result  = np.zeros(test_ids.shape[0])

vocabulary_size = len(tokenizer.word_index) + 1



X_fit = csr_matrix(train_X_padded, dtype='float32')

y_fit = train_y

gc.collect()



model = Sequential()



model.add(Embedding(input_dim=vocabulary_size, output_dim=64))

model.add(CuDNNLSTM(512, return_sequences=False))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_fit, y_fit, epochs=2, batch_size=32)







del X_fit

gc.collect()

    

test = csr_matrix(test_X_padded, dtype='float32')

test_result += model.predict_proba(test)[:,0]



del test

gc.collect()

    



submission = pd.read_csv('../input/sample_submission.csv')

submission['prediction'] = test_result

submission.to_csv('./submission.csv', index=False)