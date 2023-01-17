import numpy as np
import pandas as pd
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.head()
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


X_train, X_test = train_test_split(df, test_size=0.1, random_state=2018)

# config values
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

y_train, y_test = X_train['target'].values, X_test['target'].values

X_train = X_train['question_text'].fillna('_NA_').values
X_test = X_test['question_text'].fillna('_NA_').values
X_submission = df_test['question_text'].fillna('_NA_').values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_submission = tokenizer.texts_to_sequences(X_submission)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
X_submission = pad_sequences(X_submission, maxlen=maxlen)
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model


inp = Input(shape=(maxlen,))
layer = Embedding(max_features, embed_size)(inp)
layer = Bidirectional(LSTM(64, return_sequences=True))(layer)
layer = GlobalMaxPool1D()(layer)
layer = Dense(16, activation="relu")(layer)
layer = Dropout(0.1)(layer)
layer = Dense(1, activation="sigmoid")(layer)
model = Model(inputs=inp, outputs=layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_test, y_test))
from sklearn import metrics

pred_test_y = model.predict([X_test], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print('F1 score at threshold {} is {}'.format(thresh, metrics.f1_score(y_test, (pred_test_y > thresh).astype(int))))
pred_submission_y = model.predict([X_submission], batch_size=1024, verbose=1)
pred_submission_y = (pred_submission_y > 0.29).astype(int)

df_submission = pd.DataFrame({'qid': df_test['qid'].values})
df_submission['prediction'] = pred_submission_y
df_submission.to_csv("submission.csv", index=False)
