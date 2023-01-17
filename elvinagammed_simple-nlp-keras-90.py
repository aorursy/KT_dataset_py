# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
X = train.text
y = train.target
X
y
X = X.astype('str')
test['text'] = test['text'].astype('str')
import seaborn as sns
sns.countplot(y)
import string
import re
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
X = X.apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))
X
import nltk
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# def clean_text(df, wrong_words_dict, autocorrect=True):
#     df.fillna("__NA__", inplace=True)
#     tokinizer = RegexpTokenizer(r'\w+')
#     regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]
#     texts = df.tolist()
#     result = []
#     for text in tqdm(texts):
#         tokens = tokinizer.tokenize(text.lower())
#         tokens = [split_text_and_digits(token, regexps) for token in tokens]
#         tokens = [substitute_repeats(token, 3) for token in tokens]
#         text = ' '.join(tokens)
#         if autocorrect:
#             for wrong, right in wrong_words_dict.items():
#                 text = text.replace(wrong, right)
#         result.append(text)
#     return result 
X = X.apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
X
import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
X = X.apply(lambda x: remove_stopwords(x))
test['text'] = test['text'].apply(lambda x: remove_stopwords(x))
X
wnl = nltk.stem.WordNetLemmatizer()
def lemmatize(text):
    words = [wnl.lemmatize(w) for w in text]
    return words
X = X.apply(lambda x: lemmatize(x))
test['text'] = test['text'].apply(lambda x: lemmatize(x))
X
# Bog
MAX_NB_WORDS = 100000
max_seq_len = 30
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
tokenizer = Tokenizer(num_words=MAX_NB_WORDS+1, char_level=False)
tokenizer.fit_on_texts(pd.concat([X, test['text']]))
word_seq_train = tokenizer.texts_to_sequences(X)
word_seq_test = tokenizer.texts_to_sequences(test['text'])
word_index = tokenizer.word_index
print("dictionary size: ", len(tokenizer.word_index))

#pad sequences
# [[1], [2, 3],[4, 5, 6]] =>
# [[0, 0, 1],
#  [0, 2, 3],
#  [4, 5, 6]]
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
import codecs
from tqdm import tqdm

ft = codecs.open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec', encoding='utf-8')
embeddings_index = {}

for line in tqdm(ft):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    
ft.close()
print('found %s word vectors' % len(embeddings_index))
batch_size = 54 
num_epochs = 15

#model parameters
num_filters = 40
embed_dim = 300 
weight_decay = 1e-4
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words+1, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping
# In 1D CNN, the kernel moves in 1 direction. 
# The input and output data of 1D CNN is 2 dimensional. 
# Mostly used on Time-Series Data, Natural Language Processing tasks.

# In 2D CNN, the kernel moves in 2 directions. 
# The input and output data of 2D CNN is 3 dimensional. 
# Mostly used on Image data. 

# In 3D CNN, the kernel moves in 3 directions. 
# The input and output data of 3D CNN is 4 dimensional. 
# Mostly used on 3D Image data (MRI, CT Scans). 
model = Sequential()
model.add(Embedding(nb_words+1, embed_dim,
          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
history = model.fit(word_seq_train, y, batch_size=batch_size,
                 epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)
loss = pd.DataFrame({"loss": history.history['loss'],
                        "val_loss": history.history['val_loss']})
loss.plot()
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
target = model.predict(word_seq_test)
submission['target'] = target.round().astype(int)
submission.to_csv('submission.csv',index=False)
submission