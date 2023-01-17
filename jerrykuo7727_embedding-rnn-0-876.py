import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
train_df = pd.read_pickle("../input/hw2model/train.pkl").sample(frac=1, random_state=123)
test_df = pd.read_pickle("../input/hw2model/test.pkl")
w2v_model = Word2Vec.load("../input/hw2model/word2vec.model")
train_df.head()
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)
def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)
PADDING_LENGTH = 200
X = text_to_index(train_df.text)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
print("Shape:", X.shape)
print("Sample:", X[0])
Y = to_categorical(train_df.category)
print("Shape:", Y.shape)
print("Sample:", Y[0])
def new_model():
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(16))    
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model = new_model()
model.summary()
model.fit(x=X, y=Y, batch_size=3000, epochs=100, validation_split=0.1)
X_test = text_to_index(test_df.text)
X_test = pad_sequences(X_test, maxlen=PADDING_LENGTH)
Y_preds = model.predict(X_test)
print("Shape:", Y_preds.shape)
print("Sample:", Y_preds[0])
Y_preds_label = np.argmax(Y_preds, axis=1)
print("Shape:", Y_preds_label.shape)
print("Sample:", Y_preds_label[0])
submit = test_df[['id']]
submit['category'] = Y_preds_label
submit.to_csv("submit.csv", index=False)