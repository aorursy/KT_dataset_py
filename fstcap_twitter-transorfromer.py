!pip install tweet-preprocessor
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

import unicodedata
import re
import time
import gensim

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

import preprocessor as twitter_p

from tqdm import tqdm

import spacy
import gc

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
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(train.shape)
print(test.shape)
train.head()
test.head()
train.info()
def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return round(float(len(c)) / (len(a) + len(b) - len(c)), 4)
results_jaccard = []
for index, row in train.iterrows():
    sentence1 = row.keyword
    sentence2 = row.text
    jaccard_score = jaccard(sentence1, sentence2)
    results_jaccard.append([sentence1, sentence2, jaccard_score])
jaccard_score = pd.DataFrame(results_jaccard, columns=['keyword', 'text', 'jaccard_score'])
#sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(6, 15))
sns.set_color_codes("pastel")
sns.countplot(y="jaccard_score",data=jaccard_score, color="b")
stopwords_en = stopwords.words('english')
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
twitter_p.set_options(twitter_p.OPT.URL)
def preprocess_sentence(w):
#    w = ' '.join(map(lambda word: abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word, w.split(' ')))
    w = twitter_p.clean(w)
    w = unicode_to_ascii(w.lower().strip())
    
    w = re.sub(r"([@#])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z@#]+", " ", w)
    w = ' '.join([word for word in w.split(' ') if word not in stopwords_en])
    
    w = w.rstrip().strip()
    return w
train = train.fillna(value='')
train.info()
train['text'] = train['text'].apply(func=preprocess_sentence)
train['keyword'] = train['keyword'].apply(func=preprocess_sentence)

print(train.head(10))
test = test.fillna(value='')
test.info()
test['text'] = test['text'].apply(func=preprocess_sentence)
test['keyword'] = test['keyword'].apply(func=preprocess_sentence)
print(test.head(10))
text_list = np.stack([*train['text'], *train['keyword'], *test['text'], *test['keyword']])
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(text_list)
print(len(tokenizer.word_index))
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

glove = np.load('/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)

# fasttext = np.load('/kaggle/input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl', allow_pickle=True)

# twitter_word2vec = np.load('/kaggle/input/twitter-word2vec300d/twitter_word2vec.npy', allow_pickle=True)

# glovepath = '/kaggle/input/glovetwitter27b100dtxt/glove.twitter.27B.200d.txt'
# embeddings_index = dict()
# with open(glovepath) as f:
#     for line in f:
#       values = line.split()
#       word = values[0]
#       coefs = np.asarray(values[1:], dtype='float32')
#       embeddings_index[word] = coefs
# print('Loaded %s word vectors.' % len(embeddings_index))
# #将数据读入字符串列表。
# def read_data(filename):
#     """读取数据单词列表。"""
#     with open(filename,'r') as f:
#         data = tf.compat.as_str(f.read()).split()
#     return data

# english_corpus = read_data('/kaggle/input/english-corpus/text8')
# print("Data size",len(english_corpus),'Data_type',type(english_corpus),'Data[0:5]',english_corpus[0:5])

# total_sentences = np.stack([*train['text'],*test['text']])
# sentences = [list(gensim.utils.tokenize(s)) for s in total_sentences]
# sentences.append(english_corpus)
# gen_word2vec=gensim.models.word2vec.Word2Vec(sentences, size=200, min_count=0, iter=20, trim_rule=None)
input_vocab_size = len(tokenizer.word_index) + 3
d_model = 300
ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer("english")
len(glove.keys())
words = glove.keys()
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
def create_embedding_matrix(vectors, to_word_it, inp_vocab_size, d_m, lemma_dict):
    no_in_vocab = []
    matrix = np.random.uniform(low=-1, high=1, size=(inp_vocab_size, d_m))
    unknown_vector = np.zeros((d_m,), dtype=np.float32) - 1
    
    for key, index in to_word_it:
        
        word = key
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
        
        word = key.lower()
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
            
        word = key.upper()
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
            
        word = key.capitalize()
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
            
        word = ps.stem(key)
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
            
        word = lc.stem(key)
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
            
        word = sb.stem(key)
        try:
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
            
        try:
            word = lemma_dict[key]
            matrix[index] = vectors[word]
            continue
        except KeyError:
            ''
        
        if len(key) > 1:
            word = correction(key)
            try:
                matrix[index] = vectors[word]
                continue
            except KeyError:
                ''
        
        try:
            matrix[index] = vectors[word]
        except KeyError:
            matrix[index] = unknown_vector
            no_in_vocab.append(word)

    del vectors

    print("no_in_vocab size:", len(no_in_vocab))
    print("embedding_matrix shape:", matrix.shape)
    
    return matrix, no_in_vocab
print("Spacy NLP ...")
text_list = pd.concat([train['text'], test['text']])

print(len(tokenizer.word_index))
nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

word_dict = {}
word_index = 1
lemma_dict = {}
docs = nlp.pipe(text_list, n_threads = 2)
word_sequences = []

for doc in tqdm(docs):
    word_seq = []
    for token in doc:
        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
            word_dict[token.text] = word_index
            word_index += 1
            lemma_dict[token.text] = token.lemma_
        if token.pos_ is not "PUNCT":
            word_seq.append(word_dict[token.text])
    word_sequences.append(word_seq)
    
del docs

gc.collect()
embedding_matrix, no_in_vocab = create_embedding_matrix(glove, tokenizer.word_index.items(), input_vocab_size, d_model, lemma_dict)
del glove
def inter_section(texts, keywords, niv): 
    niv = set(niv)
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    text_tokenizer.fit_on_texts(np.stack([*texts, *keywords]))
    vocab = set(text_tokenizer.word_index.keys())
    text_in_niv = vocab.intersection(niv)
    print("vocab:", len(vocab), len(text_in_niv), len(text_in_niv)/len(vocab))
inter_section(train['text'], train['keyword'], no_in_vocab)
inter_section(test['text'], test['keyword'], no_in_vocab)
def incorrect_count(train_texts, test_texts, vocab):
    vocab = set(vocab)
    wrong_words = []
    for text in train_texts:
        intersection = set(text.split()).intersection(vocab)
        if len(intersection)>0:
            wrong_words.extend(intersection)
    
    train_ww_count = np.asarray(Counter(wrong_words).most_common())
    train_ww_count = np.concatenate([train_ww_count, np.asarray(['train']*len(train_ww_count))[:, np.newaxis]], axis=-1)
    
    wrong_words = []
    for text in test_texts:
        intersection = set(text.split()).intersection(vocab)
        if len(intersection)>0:
            wrong_words.extend(intersection)
    
    test_ww_count = np.asarray(Counter(wrong_words).most_common())
    test_ww_count = np.concatenate([test_ww_count, np.asarray(['test']*len(test_ww_count))[:, np.newaxis]], axis=-1)
    
    ww_count = np.concatenate([train_ww_count, test_ww_count], axis=0)
    ww_count = pd.DataFrame(ww_count, columns=['wrong_text', 'count','set'])
    ww_count['count'] = ww_count['count'].astype('int')
    ww_count = ww_count.sort_values(by='count', ascending=False)

    print('head:\n', ww_count.head())
    print('\n count<2:', len(ww_count[ww_count['count']<2])/len(ww_count))
    
    plt.figure(figsize=(6,36))
    sns.barplot(x='count', y='wrong_text', hue='set', orient='h', data=ww_count.head(100))
    plt.show()
incorrect_count(train['text'], test['text'], no_in_vocab)
def len_sentence(texts, key):
    result_num = []
    for index, row in texts.iterrows():
        sentence = row[key]
        num_text = len(sentence.split())
        result_num.append([row['id'], num_text, row['target']])

    num_texts = pd.DataFrame(result_num, columns=['id', 'Num_text', 'target'])
    num_texts_sort = num_texts.sort_values(by='Num_text', ascending=False)
    
    plt.figure(figsize=(6, 18)) 
    sns.countplot(y='Num_text', hue='target', data=num_texts_sort, color='b')
    plt.show()
    
    return num_texts
num_texts =len_sentence(train, 'text')
num_keyword = len_sentence(train, 'keyword')
# train = train.merge(num_texts, how='outer')
# train.info()
# train = train[train['Num_text'] <= 22]
train.info()
def texts_to_sequences(byte):
    char = str(byte, encoding='utf-8')
    sequences = tokenizer.texts_to_sequences([char])
    return np.reshape(sequences, (-1))
def text_encode(keyword, lang, target):
    keyword = texts_to_sequences(keyword.numpy())
    lang = [len(tokenizer.word_index), *keyword, len(tokenizer.word_index) + 1, *texts_to_sequences(lang.numpy()), len(tokenizer.word_index)+2]
    return lang, target
def tf_encode(id_num, keyword, lang, target):
    lang, target = tf.py_function(
        text_encode, 
        [keyword, lang, target], 
        [tf.int64, tf.int64])
    
    id_num.set_shape(None)
    lang.set_shape([None])
    target.set_shape([])
    
    return id_num, lang, target
class EmbeddingLayer(object):
    def __init__(self):
        
        self.kernels = tf.Variable(initial_value=embedding_matrix, trainable=False, name='Embedding_kernels')
    def __call__(self, x):
        embeddings = tf.nn.embedding_lookup(params=self.kernels, ids=x)
        return embeddings
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
    return pos * angle_rates
def positional_encoding(postion, d_model):
    angle_rads = get_angles(np.arange(postion)[:,np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)), -1, 0)
    return mask
def scaled_dot_product_attention(q, k, v, mask):
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(q)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights
class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(units=d_model)
        self.wk = tf.keras.layers.Dense(units=d_model)
        self.wv = tf.keras.layers.Dense(units=d_model)
        
        self.dense = tf.keras.layers.Dense(units=d_model)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=dff, activation='relu'),
        tf.keras.layers.Dense(units=d_model)
    ])
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.ffn = point_wise_feed_forward_network(d_model=d_model, dff=dff)
        
        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training, mask):
        attn_output, attn_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate=rate)
        
    def call(self, x, training, mask, positition=True):
        
        seq_len = tf.shape(x)[1]
        
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        if positition:
            x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        return x
class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, units, rate):
        super(OutputLayer, self).__init__()
        self.gapool1d = tf.keras.layers.GlobalAveragePooling1D()
        
        self.dense = tf.keras.layers.Dense(units=units, activation='relu')
        self.final_layer = tf.keras.layers.Dense(units=2)
        
        self.dropout = tf.keras.layers.Dropout(rate=rate)
    def call(self, enc, training):
        x = self.gapool1d(enc)

        x = self.dense(x)
    
        x = self.dropout(x, training=training)
    
        x = self.final_layer(x)
        return x
class TransformerCategorical(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, output_units, rate=0.1):
        super(TransformerCategorical, self).__init__()
        
        self.embedding = EmbeddingLayer()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, maximum_position_encoding, rate)
        
        self.outputlayer = OutputLayer(output_units, rate)
        
    def call(self, lang, training, enc_padding_mask):
        
        enc_input = self.embedding(lang)
        
        enc_output = self.encoder(enc_input, training, enc_padding_mask)
        
        out = self.outputlayer(enc_output, training)
        return out
num_layers = 6
d_model = d_model
num_heads = 6
dff = 512
pe_input = input_vocab_size
output_units = 64
rate = 0.1
tsfr_categorical = TransformerCategorical(num_layers, d_model, num_heads, dff, pe_input, output_units, rate)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=600):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = step + 100
        arg1 = step ** -0.8
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(300, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
a = tf.Variable([[1],[2]])
tf.squeeze(a, axis=1).numpy()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)

def acc_function(real, pred):
    predictions = tf.math.argmax(pred, axis=1)
    predictions = tf.cast(predictions, dtype=tf.int64)
    accuracy = tf.cast(tf.math.equal(predictions, real), dtype=tf.float32)
    ave_acc = tf.reduce_mean(accuracy)
    return ave_acc
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
]
@tf.function(input_signature=train_step_signature)
def train_step(lang, targ):

    enc_padding_mask = create_padding_mask(lang)
    
#     look_ahead_mask = create_look_ahead_mask(tf.shape(lang)[1])
#     dec_target_padding_mask = create_padding_mask(lang)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    with tf.GradientTape() as tape:
        predictions = tsfr_categorical(lang, True, enc_padding_mask)
        loss = loss_function(targ, predictions)
        
        loss_regularization = []
        for w in tsfr_categorical.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(w))
        
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
        
        loss = loss + 0.01 * loss_regularization
        
    gradients = tape.gradient(loss, tsfr_categorical.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tsfr_categorical.trainable_variables))
    
    accuracy = acc_function(targ, predictions)
    
    return loss, accuracy
@tf.function(input_signature=train_step_signature)
def valid_step(lang, targ):
    enc_padding_mask = create_padding_mask(lang)
    
    predictions = tsfr_categorical(lang, False, enc_padding_mask)
    loss = loss_function(targ, predictions)
    
    accuracy = acc_function(targ, predictions)
    
    return loss, accuracy
BATCH_SIZE = 2048
BUFFLE_SIZE = 8000
def data_generator(data):
    dataset = tf.data.Dataset.from_tensor_slices((data['id'], data['keyword'], data['text'], data['target']))
    dataset = dataset.map(tf_encode)
    dataset = dataset.cache().shuffle(BUFFLE_SIZE).padded_batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

train_data = train.sample(frac=0.9)
val_data = train[~train.index.isin(train_data.index)]

train_dataset = data_generator(train_data)
val_dataset = data_generator(val_data)

sample = next(iter(train_dataset))
print(sample)
Epochs = 100
tensorboard = {'Train_loss':[],'Train_acc':[],'Val_loss':[],'Val_acc':[]}
for epoch in range(Epochs):
    
    train_loss = []
    train_accuracy = []
    
    val_loss = []
    val_accuracy = []
    
    for _, lang, targ in train_dataset:
        loss, acc = train_step(lang, targ)
        train_loss.append(loss)
        train_accuracy.append(acc)
    
    for _, lang, targ in val_dataset:
        loss, acc = valid_step(lang, targ)
        val_loss.append(loss)
        val_accuracy.append(acc)
    
    ave_val_acc = np.mean(val_accuracy)
    
    if epoch !=0 and max(tensorboard['Val_acc']) <= ave_val_acc:
        tsfr_categorical.save_weights('/kaggle/working/checkpoint/best_val')
    
    tensorboard['Train_loss'].append(np.mean(train_loss))
    tensorboard['Train_acc'].append(np.mean(train_accuracy))
    tensorboard['Val_loss'].append(np.mean(val_loss))
    tensorboard['Val_acc'].append(ave_val_acc)
for index in range(len(tensorboard['Train_loss'])):
    print(f"\033[0;34mEpoch\033[0m:{index}, Loss:{tensorboard['Train_loss'][index]}, Accuracy:{tensorboard['Train_acc'][index]}",
     f"Valid_Loss:{tensorboard['Val_loss'][index]}, Valid_Accuracy:{tensorboard['Val_acc'][index]}")
tensorboard = pd.DataFrame(tensorboard, range(len(tensorboard['Train_loss'])))

plt.figure(figsize=(18,6))
sns.lineplot(data=tensorboard[['Train_loss', 'Val_loss']], palette="tab10", linewidth=2.5)

plt.figure(figsize=(18,6))
sns.lineplot(data=tensorboard[['Train_acc', 'Val_acc']], palette="tab10", linewidth=2.5)
venv_target = np.array([0]*len(test['text']))

test_target = pd.read_csv('/kaggle/input/test-twitter/perfect_submission.csv')

test_dataset = tf.data.Dataset.from_tensor_slices((test['id'], test['keyword'], test['text'], test_target['target']))
test_dataset = test_dataset.map(tf_encode)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)
evaluate = TransformerCategorical(num_layers, d_model, num_heads, dff, pe_input, output_units, rate)
evaluate.load_weights('/kaggle/working/checkpoint/best_val')
results = []

for id_num, lang, _ in test_dataset:
    
    enc_padding_mask = create_padding_mask(lang)
    
    predictions = evaluate(lang, False, enc_padding_mask)
    
    predictions = tf.math.argmax(predictions, axis=1)
    predictions = tf.cast(predictions, dtype=tf.int32)
    predictions = tf.reshape(predictions, (-1))
    
    results.extend(zip(id_num.numpy(), predictions.numpy()))
label_equal = 0
for index, value in enumerate(results):
    if value[1] == test_target['target'][index]:
        label_equal +=1
print('result_score:', label_equal/len(results))
submission = pd.DataFrame(results, columns=['id', 'target'])
print(submission.head())
print(submission.info())
print(submission.describe())
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(time.time() - start_time)
gc.collect()