import re
import os
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers, models, utils
import json
def reset_everything():
    tf.reset_default_graph()
    K.set_session(tf.InteractiveSession())
# Constants for our networks.  I keep these deliberately small to reduce training time.

VOCAB_SIZE = 250000
EMBEDDING_SIZE = 100
MAX_DOC_LEN = 128
MIN_DOC_LEN = 12
def extract_stackexchange(filename, limit=1000000):
    json_file = f'{filename}limit={limit}.json'

    rows = []
    for i, line in enumerate(os.popen(f'7z x -so "{filename}" Posts.xml')):
        line = str(line)
        if not line.startswith('  <row'):
            continue
            
        if i % 1000 == 0:
            print('\r%05d/%05d' % (i, limit), end='', flush=True)

        parts = line[6:-5].split('"')
        record = {}
        for i in range(0, len(parts), 2):
            k = parts[i].replace('=', '').strip()
            v = parts[i+1].strip()
            record[k] = v
        rows.append(record)
        
        if len(rows) > limit:
            break
    
    with open(json_file, 'w') as fout:
        json.dump(rows, fout)
    
    return rows


xml_7z = utils.get_file(
    fname='datascience.stackexchange.com.7z',
    origin='https://archive.org/download/stackexchange/datascience.stackexchange.com.7z',
)
print()

rows = extract_stackexchange(xml_7z)
df = pd.DataFrame.from_records(rows)    
df = df.set_index('Id', drop=False)
df['Title'] = df['Title'].fillna('').astype('str')
df['Tags'] = df['Tags'].fillna('').astype('str')
df['Body'] = df['Body'].fillna('').astype('str')
df['Id'] = df['Id'].astype('int')
df['PostTypeId'] = df['PostTypeId'].astype('int')
df['ViewCount'] = df['ViewCount'].astype('float')

df.head()
# Popular topics
list(df[df['ViewCount'] > 50000]['Title'])
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df['Body'] + df['Title'])
# Compute TF/IDF Values

total_count = sum(tokenizer.word_counts.values())
idf = { k: np.log(total_count/v) for (k,v) in tokenizer.word_counts.items() }
# Download pre-trained word2vec embeddings

import gensim

glove_100d = utils.get_file(
    fname='glove.6B.100d.txt',
    origin='https://storage.googleapis.com/deep-learning-cookbook/glove.6B.100d.txt',
)

w2v_100d = glove_100d + '.w2v'
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_100d, w2v_100d)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_100d)

w2v_weights = np.zeros((VOCAB_SIZE, w2v_model.syn0.shape[1]))
idf_weights = np.zeros((VOCAB_SIZE, 1))

for k, v in tokenizer.word_index.items():
    if v >= VOCAB_SIZE:
        continue
    
    if k in w2v_model:
        w2v_weights[v] = w2v_model[k]
    
    idf_weights[v] = idf[k]
    
del w2v_model
df['title_tokens'] = tokenizer.texts_to_sequences(df['Title'])
df['body_tokens'] = tokenizer.texts_to_sequences(df['Body'])
import random

# I created a data generator that will randomly title and body tokens for questions. 
# I'll use random text from other questions as a negative example when necessary.
def data_generator(batch_size, negative_samples=1):
    questions = df[df['PostTypeId'] == 1]
    all_q_ids = list(questions.index)
        
    batch_x_a = []
    batch_x_b = []
    batch_y = []
    
    def _add(x_a, x_b, y):
        batch_x_a.append(x_a[:MAX_DOC_LEN])
        batch_x_b.append(x_b[:MAX_DOC_LEN])
        batch_y.append(y)
    
    while True:
        questions = questions.sample(frac=1.0)
        
        for i, q in questions.iterrows():
            _add(q['title_tokens'], q['body_tokens'], 1)
            
            negative_q = random.sample(all_q_ids, negative_samples)
            for nq_id in negative_q:
                _add(q['title_tokens'], df.at[nq_id, 'body_tokens'], 0)            
            
            if len(batch_y) >= batch_size:
                yield ({
                    'title': pad_sequences(batch_x_a, maxlen=None),
                    'body': pad_sequences(batch_x_b, maxlen=None),
                }, np.asarray(batch_y))
                
                batch_x_a = []
                batch_x_b = []
                batch_y = []

# dg = data_generator(1, 2)
# next(dg)
# next(dg)
questions = df[df['PostTypeId'] == 1]['Title'].reset_index(drop=True)
question_tokens = pad_sequences(tokenizer.texts_to_sequences(questions))

class EmbeddingWrapper(object):
    def __init__(self, model):
        self._r = questions
        self._i = {i:s for (i, s) in enumerate(questions)}
        self._w = model.predict({'title': question_tokens}, verbose=1, batch_size=1024)
        self._model = model
        self._norm = np.sqrt(np.sum(self._w * self._w + 1e-5, axis=1))

    def nearest(self, sentence, n=10):
        x = tokenizer.texts_to_sequences([sentence])
        if len(x[0]) < MIN_DOC_LEN:
            x[0] += [0] * (MIN_DOC_LEN - len(x))
        e = self._model.predict(np.asarray(x))[0]
        norm_e = np.sqrt(np.dot(e, e))
        dist = np.dot(self._w, e) / (norm_e * self._norm)

        top_idx = np.argsort(dist)[-n:]
        return pd.DataFrame.from_records([
            {'question': self._r[i], 'dist': float(dist[i])}
            for i in top_idx
        ])
# The first model will just sum up the embeddings of each token.
# The similarity between documents will be the dot product of the final embedding.

import tensorflow as tf

def sum_model(embedding_size, vocab_size, embedding_weights=None, idf_weights=None):
    title = layers.Input(shape=(None,), dtype='int32', name='title')
    body = layers.Input(shape=(None,), dtype='int32', name='body')

    def make_embedding(name):
        if embedding_weights is not None:
            embedding = layers.Embedding(mask_zero=True, input_dim=vocab_size, output_dim=w2v_weights.shape[1], 
                                         weights=[w2v_weights], trainable=False, 
                                         name=f'{name}/embedding')
        else:
            embedding = layers.Embedding(mask_zero=True, input_dim=vocab_size, output_dim=embedding_size,
                                        name=f'{name}/embedding')

        if idf_weights is not None:
            idf = layers.Embedding(mask_zero=True, input_dim=vocab_size, output_dim=1, 
                                   weights=[idf_weights], trainable=False,
                                   name=f'{name}/idf')
        else:
            idf = layers.Embedding(mask_zero=True, input_dim=vocab_size, output_dim=1,
                                   name=f'{name}/idf')
            
        return embedding, idf
    
    embedding_a, idf_a = make_embedding('a')
    embedding_b, idf_b = embedding_a, idf_a
#     embedding_b, idf_b = make_embedding('b')

    mask = layers.Masking(mask_value=0)
    def _combine_and_sum(args):
        [embedding, idf] = args
        return K.sum(embedding * K.abs(idf), axis=1)

    sum_layer = layers.Lambda(_combine_and_sum, name='combine_and_sum')

    sum_a = sum_layer([mask(embedding_a(title)), idf_a(title)])
    sum_b = sum_layer([mask(embedding_b(body)), idf_b(body)])

    sim = layers.dot([sum_a, sum_b], axes=1, normalize=True)
    sim_model = models.Model(
        inputs=[title, body],
        outputs=[sim],
    )
    sim_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    sim_model.summary()

    embedding_model = models.Model(
        inputs=[title],
        outputs=[sum_a]
    )
    return sim_model, embedding_model
# Try using our model with pretrained weights from word2vec

# Uncomment to run
# sum_model_precomputed, sum_embedding_precomputed = sum_model(
#     embedding_size=EMBEDDING_SIZE, vocab_size=VOCAB_SIZE,
#     embedding_weights=w2v_weights, idf_weights=idf_weights
# )

# x, y = next(data_generator(batch_size=4096))
# sum_model_precomputed.evaluate(x, y)
SAMPLE_QUESTIONS = [
    'What is an activation function',
    'Best techniques in computer vision',
    'Sentiment analysis in twitter',
]

def evaluate_sample(lookup):
    pd.set_option('display.max_colwidth', 100)
    results = []
    for q in SAMPLE_QUESTIONS:
        print(q)
        q_res = lookup.nearest(q, n=4)
        q_res['result'] = q_res['question']
        q_res['question'] = q
        results.append(q_res)

    return pd.concat(results)

lookup = EmbeddingWrapper(model=sum_embedding_precomputed)
evaluate_sample(lookup)
sum_model_trained, sum_embedding_trained = sum_model(
    embedding_size=EMBEDDING_SIZE, vocab_size=VOCAB_SIZE, 
    embedding_weights=None,
    idf_weights=None
)

# Uncomment to run
# sum_model_trained.fit_generator(
#     data_generator(batch_size=128),
#     epochs=10,
#     steps_per_epoch=1000
# )
lookup = EmbeddingWrapper(model=sum_embedding_trained)
evaluate_sample(lookup)