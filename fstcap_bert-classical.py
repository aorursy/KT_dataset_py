!pip install -U pip

!pip install tweet-preprocessor

!pip install -q tf-nightly

!pip install -q tf-models-nightly
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf



from nltk.corpus import stopwords

from nltk.stem.lancaster import LancasterStemmer

from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer

ps = PorterStemmer()

lc = LancasterStemmer()

sb = SnowballStemmer("english")



import preprocessor as twitter_p



from official.nlp import bert

from official import nlp



import official.nlp.bert.tokenization

import official.nlp.bert.configs

import official.nlp.bert.bert_models

import official.nlp.optimization



import time

import unicodedata

import re

import json

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tf.random.set_seed(123)

np.random.seed(123)

start_time = time.time()
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
stopwords_en = stopwords.words('english')



def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



twitter_p.set_options(twitter_p.OPT.URL)

def preprocess_sentence(w):

    w = twitter_p.clean(w)

    w = unicode_to_ascii(w.lower().strip())

    

    w = re.sub(r"([@#])", r" \1 ", w)

    w = re.sub(r'[" "]+', " ", w)

#    w = re.sub(r"[^a-zA-Z@#]+", " ", w)

    w = ' '.join([word for word in w.split(' ') if word not in stopwords_en])

    w = w.rstrip().strip()

    return w
data = data.fillna(value='')

data['text'] = data['text'].apply(func=preprocess_sentence)

data['keyword'] = data['keyword'].apply(func=preprocess_sentence)



test = test.fillna(value='')

test['text'] = test['text'].apply(func=preprocess_sentence)

test['keyword'] = test['keyword'].apply(func=preprocess_sentence)
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"

tf.io.gfile.listdir(gs_folder_bert)
tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(gs_folder_bert, 'vocab.txt'), do_lower_case=True)

print('Vocab Size:', len(tokenizer.vocab))
words = tokenizer.vocab

w_rank = {}

for i,word in enumerate(words):

    w_rank[word] = i

WORDS = w_rank



def words(text): return re.findall(r'\w+', text.lower())

def P(word): 

    "Probability of `word`."

    # use inverse of rank as proxy

    # returns 0 if the word isn't in the dictionary

    return - WORDS.get(word, 0)

def correction(word): 

    "Most probable spelling correction for word."

    return max(candidates(word), key=P)

def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or [word])

def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)

def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)
def search_word(key):

    word = key

    if word in tokenizer.vocab:

        return word

    

    word = ps.stem(key)

    if word in tokenizer.vocab:

        return word

    

    word = lc.stem(key)

    if word in tokenizer.vocab:

        return word

    

    word = sb.stem(key)

    if word in tokenizer.vocab:

        return word

    

    if len(key) > 1:

        word = correction(key)

        if word in tokenizer.vocab:

            return word

    return word
def encode_sentence(s, tokenizer):

    tokens = tokenizer.tokenize(s)

    tokens = list(map(search_word, tokens))

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)



def bert_encode(glue_dict, tokenizer):

    sentence1 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(glue_dict['keyword'])])

    sentence2 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(glue_dict['text'])])

    

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]

    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    

    type_cls = tf.ones_like(cls)

    type_s1 = tf.ones_like(sentence1)

    type_s2 = tf.ones_like(sentence2)

    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    

    inputs = {

        "input_word_ids": input_word_ids.to_tensor(),

        "input_mask": input_mask,

        "input_type_ids": input_type_ids

    }

    return inputs
train = data.sample(frac=0.9)

val = data[~data.index.isin(train.index)]
glue_train = bert_encode(train, tokenizer)

glue_train_labels = train['target']



glue_val = bert_encode(val, tokenizer)

glue_val_labels = val['target']



glue_test = bert_encode(test, tokenizer)
for key, value in glue_train.items():

    print(f'{key:15s} shape: {value.shape}')

print(f'glue_train_labels shape: {glue_train_labels.shape}')
bert_config_file = os.path.join(gs_folder_bert, 'bert_config.json')

config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())



bert_config = bert.configs.BertConfig.from_dict(config_dict)

print(config_dict)
bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)
checkpoint = tf.train.Checkpoint(model=bert_encoder)

checkpoint.restore(os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
epochs = 10

batch_size = 32

eval_batch_size = 32



train_data_size = len(glue_train_labels)

steps_per_epoch = int(train_data_size / batch_size)

num_train_steps = steps_per_epoch * epochs

warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
checkpoint_filepath = '/kaggle/working/checkpoint/best_val'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_filepath,

    save_weights_only=True,

    monitor='val_accuracy',

    mode='max',

    save_best_only=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
bert_classifier.fit(glue_train, glue_train_labels, validation_data=(glue_val, glue_val_labels), batch_size=32, epochs=epochs, callbacks=[checkpoint_callback])
bert_classifier.load_weights(checkpoint_filepath)

test_target = pd.read_csv('/kaggle/input/test-twitter/perfect_submission.csv')
results = bert_classifier.evaluate(glue_test, test_target['target'])
print(results)
predictions = bert_classifier.predict(glue_test)
predictions = np.argmax(predictions, axis=1)
submission = pd.DataFrame({"id":test["id"], "target":predictions})

submission.to_csv('/kaggle/working/submission.csv', index=False)