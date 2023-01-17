import numpy as np

import pandas as pd

data = pd.read_csv('/kaggle/input/text-classification-tasks/gop.csv')

data = data[['text','sentiment']]

data[data['sentiment'] == 'Negative'] = data[data['sentiment'] == 'Negative'][:2236]

data = data.dropna()
alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

char_dict = {}

for i, char in enumerate(alphabet):

    char_dict[char] = i + 1
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



def format_data(data, seq_length=15, step=5):

    data = data[data.sentiment != "Neutral"]

    data = data.sample(frac=1).reset_index(drop=True)

    data['text'] = data['text'].str.lower()



    Y = to_numerical(data['sentiment'].values) # 0: Negative; 1: Positive

    X = data['text']



    remove_rt_url(X)



    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

    tokenizer.word_index = char_dict

    tokenizer.word_index[tokenizer.oov_token] = 0



    X = tokenizer.texts_to_sequences(X)

    

    new_X, new_Y = [], []

    for i, x in enumerate(X):

        if len(x) == seq_length:

            new_X.append(x)

            new_Y.append(Y[i])

            continue

        elif len(x) < seq_length:

            continue



        for j in range(0, len(x) - seq_length - 1, step):

            new_X.append(list(x[j:j+seq_length]))

            new_Y.append(Y[i])



    return np.array(new_X), np.array(new_Y), tokenizer





def to_numerical(d):

    """Converts the categorical df[col] to numerical"""

    _, d = np.unique(d, return_inverse=True)

    return d





def remove_rt_url(df):

    url = r'((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)'

    df.replace(regex=True, inplace=True, to_replace=r'^RT ', value=r'')

    df.replace(regex=True, inplace=True, to_replace=url, value=r'')
seq_length, step = 15, 3

X, Y, tokenizer = format_data(data.copy(), seq_length, step)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
np.unique(Y_train, return_counts=True)
from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, LSTM

from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model, Sequential



# Input shape

inp = Input(shape=(seq_length,))



# Embedding and LSTM

x = Embedding(len(tokenizer.word_index), 125)(inp)

x = SpatialDropout1D(0.33)(x)

x = Bidirectional(LSTM(75, return_sequences=True))(x)



# Pooling

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([avg_pool, max_pool])



# Output layer

output = Dense(1, activation='sigmoid')(conc)



model = Model(inputs=inp, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# model.load_weights('Weights/gru5.h5')

model.fit(X_train, Y_train, epochs=25, batch_size=32, verbose=1)
model.save('model.h5')
#model = Sequential()



#model.add(Embedding(len(tokenizer.word_index), 125, input_length=X.shape[1]))

#model.add(Dropout(0.1))

#model.add(LSTM(75, dropout_U=0.2, dropout_W=0.2))

#model.add(Dense(1, activation='sigmoid'))



#model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

#model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)



#preds = model.predict_classes(X_test, batch_size=1, verbose=1)
results = model.predict(X_test, batch_size=1, verbose=1)
def convert_to_preds(results):

    """Converts probabilistic results in [0, 1] to

    binary values, 0 and 1."""

    return [1 if r > 0.5 else 0 for r in results]



preds = convert_to_preds(results)
def accuracy_percentile(preds, Y_validate):

    """Return the percentage of correct predictions for each class and in total"""

    pos_correct, neg_correct, total_correct = 0, 0, 0

    _, (neg_count, pos_count) = np.unique(Y_validate, return_counts=True)



    for i, r in enumerate(preds):

        if r == Y_validate[i]:

            total_correct += 1

            if r == 0:

                neg_correct += 1

            else:

                pos_correct += 1



    print('Positive Accuracy:', pos_correct/pos_count * 100, '%')

    print('Negative Accuracy:', neg_correct/neg_count * 100, '%')

    print('Total Accuracy:', total_correct/(pos_count + neg_count) * 100, '%')
accuracy_percentile(preds, Y_test)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score



print('AUC: {}'.format(roc_auc_score(preds, Y_test)))

print('Accuracy: {}'.format(accuracy_score(preds, Y_test)))

print('Precision: {}'.format(precision_score(preds, Y_test)))

print('Recall: {}'.format(recall_score(preds, Y_test)))

print('F1: {}'.format(f1_score(preds, Y_test)))