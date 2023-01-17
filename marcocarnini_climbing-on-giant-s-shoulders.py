import numpy as np
import pandas as pd

df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.head()
import re
import string

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stops = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_question(question):
    question = question.translate(string.punctuation)
    
    words = question.lower().split()
    question = [w for w in words if w not in stops and len(w) >= 3]
    question = ' '.join(words)
    
    question = re.sub(r'[^A-Za-z0-9^,!.\/\'+-=]', ' ', question)
    question = re.sub(r'what\'s', 'what is', question)
    question = re.sub(r'\'s', ' ', question)
    question = re.sub(r'\'ve', ' have', question)
    question = re.sub(r'n\'t', ' not', question)
    question = re.sub(r'i\'m', 'i am', question)
    question = re.sub(r'\'re', ' are', question)
    question = re.sub(r'\'d', ' would', question)
    question = re.sub(r'\'ll', ' will', question)
    
    # remove morphological affixes
    words = question.split()
    stemmed_words = [stemmer.stem(w) for w in words]
    question = ' '.join(stemmed_words)
    
    return question

df['question_text'] = df['question_text'].map(lambda q: clean_question(q))
df_test['question_text'] = df_test['question_text'].map(lambda q: clean_question(q))
df.head()
# loading embedding: https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

'Loaded %s word vectors' % len(embeddings_index)
embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in tokenizer.word_index.items():
    if index <= vocabulary_size - 1:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['question_text'].append(df_test['question_text']))

sequences = tokenizer.texts_to_sequences(df['question_text'])
padded_data = pad_sequences(sequences, maxlen=50)

padded_data.shape
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dropout, Conv1D, MaxPooling1D, LSTM, Dense


model = Sequential()
model.add(Embedding(vocabulary_size, 300, input_length=50,
                    weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
labels = df['target']
model.fit(padded_data, np.array(labels), validation_split=0.4, epochs=3)
sequences = tokenizer.texts_to_sequences(df_test['question_text'])
padded_test = pad_sequences(sequences, maxlen=50)

padded_test.shape
predicted = model.predict(padded_test, batch_size=1024, verbose=1)
sample = df.sample(int(len(df) * 0.2))
sample_label = sample['target']

sample_sequences = tokenizer.texts_to_sequences(sample['question_text'])
padded_sample = pad_sequences(sample_sequences, maxlen=50)

padded_sample.shape
predicted_sample = model.predict(padded_sample, batch_size=1024, verbose=1)
from sklearn.metrics import f1_score


for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print('F1 score at threshold {} is {}'.format(thresh, f1_score(sample_label,
                                                             (predicted_sample > thresh).astype(int))))
output = (predicted > 0.31).astype(int)
output
df_test['prediction'] = output
submission = df_test.drop(columns=['question_text'])
submission.head()
submission.to_csv('submission.csv', index=False)