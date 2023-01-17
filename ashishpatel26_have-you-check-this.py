import pandas as pd
import re
def load_training_data():
    data_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
    x = data_df['Phrase'].values
    y = data_df['Sentiment'].values
    print('training data\'s len:', x.shape[0])
    return x, y
def load_testing_data():
    data_df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
    x = data_df['Phrase'].values
    print('testing data\'s len:', x.shape[0])
    return x
x_train, y_train = load_training_data()
x_test = load_testing_data()
print(x_train[:5])
print(y_train[:5])
print(x_test[:5])
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train_seqs = tokenizer.texts_to_sequences(list(x_train))
print(x_train_seqs[:5])
word2idx = tokenizer.word_index
from keras.preprocessing.sequence import pad_sequences
x_train_paded = pad_sequences(x_train_seqs)
print(x_train_paded.shape)
print(x_train_paded[:5])
from keras.utils import to_categorical
y_train_onehot = to_categorical(y_train)
print(y_train_onehot.shape)
print(y_train_onehot[:5])
import numpy as np
def shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]
x_train_shuffled, y_train_shuffled = shuffle(x_train_paded, 
                                             y_train_onehot)
print(x_train_shuffled[:5])
print(y_train_shuffled[:5])
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('word2vec.6B.100d.txt')
embeddings = np.zeros((len(word2idx) + 1, 100))
'the' in wv.vocab
for word, idx in word2idx.items():
    if word in wv.vocab:
        embeddings[idx] = wv.get_vector(word)
print(embeddings[:5])
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Activation
gru_model = Sequential()
gru_model.add(Embedding(embeddings.shape[0], 
                        100, 
                        weights=[embeddings], 
                        trainable=False))
gru_model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
gru_model.add(Dense(5, activation='softmax'))
gru_model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
gru_model.fit(x_train_shuffled, y_train_shuffled, batch_size=256, 
              epochs=10, verbose=1)
x_test_seqs = tokenizer.texts_to_sequences(x_test)
x_test_paded = pad_sequences(x_test_seqs)
test_pred = gru_model.predict_classes(x_test_paded)
print(test_pred)
test_df = pd.read_csv('test.tsv', sep='\t')
test_df['Sentiment'] = test_pred.reshape(-1, 1)
test_df.to_csv('gru-word2vec.csv', columns=['PhraseId', 'Sentiment'], 
               index=False, header=True)
