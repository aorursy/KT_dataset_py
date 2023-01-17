import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
# Visual
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
y = train.target
X = train.text
test['text']
X = X.astype('str')
test['text'] = test['text'].astype('str')
g = sns.countplot(y)
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
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
X.head()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
X = X.apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
X.head()
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


X = X.apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))
X.head()
# ps = nltk.stem.PorterStemmer()

# def stemmer(text):
#     words = [ps.stem(w) for w in text]
#     return words
# X = X.apply(lambda x : stemmer(x))
# test['text'] = test['text'].apply(lambda x : stemmer(x))
# X.head()
wnl = nltk.stem.WordNetLemmatizer()

def lemmatizer(text):
    words = [wnl.lemmatize(w) for w in text]
    return words
X = X.apply(lambda x : lemmatizer(x))
test['text'] = test['text'].apply(lambda x : lemmatizer(x))
X.head()
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
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
import codecs
from tqdm import tqdm

ft = codecs.open('../input/fasttext-wikinews/wiki-news-300d-1M.vec', encoding='utf-8')
embeddings_index = {}

for line in tqdm(ft):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    
ft.close()
print('found %s word vectors' % len(embeddings_index))
#training params
batch_size = 54 
num_epochs = 20

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
#model training
hist = model.fit(word_seq_train, y, batch_size=batch_size,
                 epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
target = model.predict(word_seq_test)
submission['target'] = target.round().astype(int)
submission.to_csv('submission.csv',index=False)
submission.head()