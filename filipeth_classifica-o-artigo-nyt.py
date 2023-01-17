import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.layers import Dropout, GlobalMaxPool1D



import re

import os

from nltk.corpus import stopwords

from nltk import word_tokenize

STOPWORDS = set(stopwords.words('english'))

%matplotlib inline

import warnings 

warnings.filterwarnings('ignore')
curr_dir = '../input/'

all_headlines = []

all_articles = pd.DataFrame()

for filename in os.listdir(curr_dir):

    if 'Articles' in filename:

        article_df = pd.read_csv(curr_dir + filename)

        all_articles = pd.concat([all_articles, article_df], ignore_index=True)
all_articles.info()
all_articles.typeOfMaterial.value_counts()
df_news = teste = all_articles.loc[(all_articles['typeOfMaterial'] == 'News')][:500]

df_OpEd = teste = all_articles.loc[(all_articles['typeOfMaterial'] == 'Op-Ed')][:500]

df_review = teste = all_articles.loc[(all_articles['typeOfMaterial'] == 'Review')][:500]
all_articles = pd.concat([df_news, df_OpEd, df_review], ignore_index=True)

all_articles.shape
all_articles.isnull().sum()
print(all_articles.shape)

all_articles.drop(['abstract'], axis=1, inplace=True)

print(all_articles.shape)
all_articles['snippet'][:10]
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[/(){}\[\]\|@,;]', ' ', text) 

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text
all_articles['snippet'] = all_articles['snippet'].apply(clean_text)
tokenizer = Tokenizer(lower=True)

tokenizer.fit_on_texts(all_articles['snippet'].values)

word_index = tokenizer.word_index

print('Foram encontrados {} tokens.'.format(len(word_index)))
train_test_df = all_articles[['snippet', 'typeOfMaterial']].copy()
# Save cleaned dataframe to csv

os.chdir(r'../working')

train_test_df.to_csv(r'train_test_df.csv')
from IPython.display import FileLink

FileLink(r'train_test_df.csv')
X = tokenizer.texts_to_sequences(train_test_df['snippet'].values)

maxlen = 0

for item in X:

    if len(item) > maxlen:

        maxlen = len(item)

X = pad_sequences(X, maxlen=maxlen)

print('Tamanho input:', X.shape)
# Normaliza os valores para ficar entre 0 e 1

# X = X/len(word_index)
Y = pd.get_dummies(train_test_df['typeOfMaterial']).values

print('Tamanho output:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model_cnn = Sequential()

model_cnn.add(Embedding(len(word_index)+1, 100, input_length=X.shape[1]))

model_cnn.add(SpatialDropout1D(0.2))



model_cnn.add(Conv1D(100, 3, padding='valid', activation='relu', strides=1))

model_cnn.add(GlobalMaxPool1D())



model_cnn.add(Dropout(0.2))

model_cnn.add(Dense(3, activation='softmax'))



model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_cnn.summary())
model_lstm = Sequential()

model_lstm.add(Embedding(len(word_index)+1, 100, input_length=X.shape[1]))

model_lstm.add(SpatialDropout1D(0.2))



model_lstm.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model_lstm.add(Dropout(0.2))



model_lstm.add(Dense(3, activation='softmax'))

model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_lstm.summary())



# model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))

# model.add(LSTM(100))

# model.add(Dense(32, activation='relu'))

# model.add(Dropout(0.2))



# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
epochs = 10

batch_size = 64



history_cnn = model_cnn.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)




history_lstm = model_lstm.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
accr_cnn = model_cnn.evaluate(X_test,Y_test)

print('CNN  Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr_cnn[0],accr_cnn[1]))



accr_lstm = model_lstm.evaluate(X_test,Y_test)

print('LSTM  Loss: {:0.3f}  Accuracy: {:0.3f}'.format(accr_lstm[0],accr_lstm[1]))
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import time

def model_result_info(model, X_test, y_test):

    start = time.time()

    y_pred1 = model.predict(X_test)

    time_predict = time.time() - start

    y_pred = np.argmax(y_pred1, axis=1)

    y_test1 = np.argmax(y_test, axis=1)

    loss, accuracy = model.evaluate(X_test,y_test, verbose=0)

    # Print f1, precision, and recall scores

    precision = precision_score(y_test1, y_pred, average='macro')

    recall = recall_score(y_test1, y_pred, average='macro')

    f1 = f1_score(y_test1, y_pred, average='macro')

    return loss, accuracy, f1, precision, recall, time_predict
# evaluate the models

loss_cnn, accuracy_cnn, f1_score_cnn, precision_cnn, recall_cnn, time_predict_cnn = model_result_info(model_cnn, X_test, Y_test)

print("CNN - loss:{:0.2f}, accuracy:{:0.2f}%, f1_score:{:0.2f}%, precision:{:0.2f}%, recall:{:0.2f}%".format(loss_cnn, accuracy_cnn*100, f1_score_cnn*100, precision_cnn*100, recall_cnn*100))

loss_lstm, accuracy_lstm, f1_score_lstm, precision_lstm, recall_lstm, time_predict_lstm = model_result_info(model_lstm, X_test, Y_test)

print("LSMT- loss:{:0.2f}, accuracy:{:0.2f}%, f1_score:{:0.2f}%, precision:{:0.2f}%, recall:{:0.2f}%".format(loss_lstm, accuracy_lstm*100, f1_score_lstm*100, precision_lstm*100, recall_lstm*100))
print("Tempo em segundos para predicao de {} artigos - CNN: {:0.3f} e LSTM:{:0.3f}".format(len(X_test), time_predict_cnn, time_predict_lstm))
plt.title('Loss CNN')

plt.plot(history_cnn.history['loss'], label='train')

plt.plot(history_cnn.history['val_loss'], label='test')

plt.legend()

plt.show();

plt.title('Loss LSTM')

plt.plot(history_lstm.history['loss'], label='train')

plt.plot(history_lstm.history['val_loss'], label='test')

plt.legend()

plt.show();
plt.title('Accuracy CNN')

plt.plot(history_cnn.history['acc'], label='train')

plt.plot(history_cnn.history['val_acc'], label='test')

plt.legend()

plt.show();



plt.title('Accuracy LSTM')

plt.plot(history_lstm.history['acc'], label='train')

plt.plot(history_lstm.history['val_acc'], label='test')

plt.legend()

plt.show();
import sklearn.metrics as metrics

y_true_labels = np.argmax(Y_test, axis=1)





y_pred_ohe = model_cnn.predict(X_test)

y_pred_labels = np.argmax(y_pred_ohe, axis=1)

confusion_matrix_cnn = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)



y_pred_ohe = model_lstm.predict(X_test)

y_pred_labels = np.argmax(y_pred_ohe, axis=1)

confusion_matrix_lstm = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
plt.imshow(confusion_matrix_cnn)

plt.xlabel("Predicted labels")

plt.ylabel("True labels")

plt.xticks([], [])

plt.yticks([], [])

plt.title('Confusion matrix CNN')

plt.colorbar()

plt.show()



plt.imshow(confusion_matrix_lstm)

plt.xlabel("Predicted labels")

plt.ylabel("True labels")

plt.xticks([], [])

plt.yticks([], [])

plt.title('Confusion matrix LSTM')

plt.colorbar()

plt.show()