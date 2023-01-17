# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from tqdm import tqdm
tqdm.pandas()
np.random.seed(1)
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))



# train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
# sentences = train_df["question_text"].apply(lambda x: x.split())
# vocab = build_vocab(sentences)
#oov = check_coverage(vocab,embeddings_index)



train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)
def get_synonyms(word, pos=None):
    wordnet_pos = {
        "NN": wordnet.NOUN,
        "VB": wordnet.VERB,
        "VBD": wordnet.VERB,
        "VBG": wordnet.VERB,
        "VBN": wordnet.VERB,
        "VBP": wordnet.VERB,
        "JJ": wordnet.ADJ,
        "RB": wordnet.ADV,
        "RBR": wordnet.ADV,
        "RBS": wordnet.ADV,
    }
    if pos:
        if pos in list(wordnet_pos.keys()):
            synsets = wordnet.synsets(word, pos=wordnet_pos[pos])
            synonyms = []
            for synset in synsets:
                synonyms += [str(lemma.name()) for lemma in synset.lemmas()]
            synonyms = [synonym.replace("_", " ") for synonym in synonyms]
            synonyms = list(set(synonyms))
            synonyms = [synonym for synonym in synonyms if synonym != word]
            if synonyms:
                return synonyms[0]
    return ''


def get_syn_sentence(text):
    words = text.split()
    words_with_pos_tag = pos_tag(words)
    words_with_pos_tag
    new_sentence_words = []
    for word, pos in words_with_pos_tag:
        synonym = get_synonyms(word, pos)
        if synonym:
            new_sentence_words.append(synonym)
        else:
            new_sentence_words.append(word)
    synonym_sentence = ' '.join(new_sentence_words)
    return synonym_sentence
#df_obscene = train_df.loc[train_df['column_name'] == some_value]
df_obscene = train_df[train_df['target'] == 1]
df_obscene = df_obscene.reset_index()

sentence_with_synonyms = []
for idx, row in df_obscene.iterrows():
    sentence_with_synonyms.append(get_syn_sentence(row['question_text']))
                             
df_obscene['question_text'] = sentence_with_synonyms


    

#df_obscene.head()
train_df = train_df.append(df_obscene, ignore_index=True, sort = False)
#train_df.head()
## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = LSTM(128, return_sequences = True)(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)