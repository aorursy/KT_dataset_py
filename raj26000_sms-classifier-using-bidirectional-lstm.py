import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', delimiter=',', encoding='latin-1')
print(df)

Y = df['v1']
X = df['v2']

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
X_train = X_train.tolist()
X_test = X_test.tolist()

X_train = [text.lower() for text in X_train]
X_test = [text.lower() for text in X_test]

label2idx = {
    'ham':0,
    'spam':1
}
maxlen = 150
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
train_seq = tokenizer.texts_to_sequences(X_train)
train_pad = pad_sequences(train_seq, maxlen=maxlen, truncating='post')
test_seq = tokenizer.texts_to_sequences(X_test)
test_pad = pad_sequences(test_seq, maxlen=maxlen, truncating='post')

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_pad, y_train, epochs=10, validation_data=(test_pad, y_test))
model.summary()
model.evaluate(test_pad, y_test)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel('epochs')
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'loss')
plot_graphs(history, 'accuracy')
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.5)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
clf = LogisticRegression(penalty='l2', C=10).fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)
print(y_pred[:1000])
y_pred_labels = []

#Converting output of classifier back in terms of input ham/spam labels.
for i in y_pred:
    for label, idx in label2idx.items():
        if i==idx:
            y_pred_labels = np.append(y_pred_labels, label)
print(y_pred_labels[:1000])

        
    

#Final accuracy on testing.
print(accuracy_score(y_test, y_pred))