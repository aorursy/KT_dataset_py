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
print(os.listdir("../input/embeddings"))
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(text):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = text.replace("  ", " ")
    for punct in puncts:
        text = text.replace(punct, f" {punct} ")
    return text
df = pd.read_csv("../input/train.csv")
df.head(1)
df = df.drop(["qid"], axis=1)
df.head(1)
df = df.dropna()
embed_size = 300 # 300 dim vector
max_features = None
maxlen = 150 # quora has a hard limit
X, Y = df["question_text"].values, df["target"].values
import pickle as pkl
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
if os.path.exists("./tokenizer.pkl"):
    with open("./tokenizer.pkl", "rb") as f:
        tokenizer = pkl.load(f)
else:
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))
    with open('./tokenizer.pkl', 'wb') as f:
        pkl.dump(tokenizer, f, protocol=pkl.HIGHEST_PROTOCOL)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=maxlen)
word_index = tokenizer.word_index
max_features = len(word_index)+1
import gc
del df
gc.collect()
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix
embedding_matrix = load_glove(word_index)
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=512, epochs=2, validation_split=0.2)
model.save("model.h5")
del X
del Y
gc.collect()
test_df = pd.read_csv("../input/test.csv")
qids = test_df["qid"].values
testX = test_df["question_text"].values
testX = tokenizer.texts_to_sequences(testX)
testX = pad_sequences(testX, maxlen=maxlen)
testY = model.predict(testX)
testY = testY.tolist()
for i in range(len(testY)):
    testY[i] = round(testY[i][0])
df = pd.DataFrame(columns=["qid", "prediction"])
df["qid"] = qids
df["prediction"] = testY
df.to_csv("submission.csv", index=False)