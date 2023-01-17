import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("ggplot")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
print('Processing text dataset')
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
# replace urls
re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                    .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                    re.MULTILINE|re.UNICODE)
# replace ips
re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

# setup tokenizer
tokenizer = WordPunctTokenizer()

vocab = Counter()

def text_to_wordlist(text, lower=False):
    # replace URLs
    text = re_url.sub("URL", text)
    
    # replace IPs
    text = re_ip.sub("IPADDRESS", text)
    
    # Tokenize
    text = tokenizer.tokenize(text)
    
    # optional: lower case
    if lower:
        text = [t.lower() for t in text]
    
    # Return a list of words
    vocab.update(text)
    return text

def process_comments(list_sentences, lower=False):
    comments = []
    for text in tqdm(list_sentences):
        txt = text_to_wordlist(text, lower=lower)
        comments.append(txt)
    return comments


list_sentences_train = list(train["question_text"].fillna("NAN_WORD").values)
list_sentences_test = list(test["question_text"].fillna("NAN_WORD").values)

comments = process_comments(list_sentences_train + list_sentences_test, lower=True)

print("The vocabulary contains {} unique tokens".format(len(vocab)))
from gensim.models import Word2Vec
model = Word2Vec(comments, size=100, window=5, min_count=3, workers=16, sg=0, negative=5)

word_vectors = model.wv

print("Number of word vectors: {}".format(len(word_vectors.vocab)))

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
#Initialize The Embeddings In Keras

MAX_NB_WORDS = len(word_vectors.vocab) #52199
MAX_SEQUENCE_LENGTH = 20
from keras.preprocessing.sequence import pad_sequences

word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
sequences = [[word_index.get(t, 0) for t in comment]
             for comment in comments[:len(list_sentences_train)]]
test_sequences = [[word_index.get(t, 0)  for t in comment] 
                  for comment in comments[len(list_sentences_train):]]
# word index -> word to number dictionary
#sequence -> array of words to array of numbered indexes
# pad
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, 
                     padding="pre", truncating="post")
y = train['target'].values
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre",
                          truncating="post")
print('Shape of test_data tensor:', test_data.shape)
#create the embedding matrix

WV_DIM = 100
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass        
#Setup The Comment Classifier

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
wv_matrix.shape
from gensim.models import KeyedVectors
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
#create the embedding matrix

WV_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
# we initialize the matrix with random numbers
wv_matrix_glove = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        #embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix_glove[i] = embeddings_index.get_vector(word)
    except:
        pass        
wv_matrix = np.concatenate((wv_matrix,wv_matrix_glove), axis=1)
WV_DIM = 400
wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
# max words = 52199,vector dim = 100,words(52199)*vectors(100),200,
# Inputs
comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = wv_layer(comment_input)
# biGRU
embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)
# Output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
preds = Dense(1, activation='sigmoid')(x)

# build the model
model = Model(inputs=[comment_input], outputs=preds)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit([data], y, validation_split=0.1,
                 epochs=5, batch_size=256, shuffle=True)
y_valpred = model.predict(data)
f1 = 0
threshold = 0
from sklearn import metrics
for thresh in np.arange(0.1, 0.501,0.01):
    f1score = np.round(metrics.f1_score(train.target, (y_valpred>thresh).astype(int)), 4)
    thresh = np.round(thresh,2)
    print('F1 score for threshold {} : {}'.format(thresh, f1score))
    if f1score > f1:
        f1 = f1score
        threshold = thresh
        print('In {} : {}'.format(threshold,f1score))
y_valpred_test = model.predict(test_data)
y_test = (y_valpred_test[:,0] > threshold).astype(np.int)
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_test})
submit_df.to_csv("submission.csv", index=False)
