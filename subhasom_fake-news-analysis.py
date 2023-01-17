import pandas as pd



d_fake = pd.read_csv('../input/fake-news-data/fnn_politics_fake.csv')

headlines_fake = d_fake.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})

headlines_fake['fake'] = 1



d_real = pd.read_csv('../input/fake-news-data/fnn_politics_real.csv')

headlines_real = d_real.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})

headlines_real['fake'] = 0



eval_data = pd.concat([headlines_fake, headlines_real])
import os



def read_data(d):

    files = os.listdir(d)

    headlines, contents = [], []

    for fname in files:

        if fname[:5] != 'polit':

            continue

        

        f = open(d + '/' + fname)

        text = f.readlines()

        f.close()



        if len(text) == 2:

            # One of the lines is missing

            if len(text[1]) <= 1:

                # There is no article content or headline

                continue

        elif len(text) >= 3:

            # More than one empty line encountered

            text[1] = text[-1]

        else:

            # Only one or zero lines is file

            continue

        

        headline, content = text[0][:-1].strip().rstrip(), text[1][:-1]

        headlines.append(headline)

        contents.append(content)

    

    return headlines, contents





fake_dir = '../input/fake-news-data/fnd_news_fake'

fake_headlines, fake_content = read_data(fake_dir)

fake_headlines = pd.DataFrame(fake_headlines, columns=['headline'])

fake_headlines['fake'] = 1



real_dir = '../input/fake-news-data/fnd_news_real'

real_headlines, real_content = read_data(real_dir)

real_headlines = pd.DataFrame(real_headlines, columns=['headline'])

real_headlines['fake'] = 0
eval_data = pd.concat([eval_data, fake_headlines, real_headlines])

eval_data['fake'].value_counts()

eval_data.head()
all_news = pd.read_csv('../input/all-the-news/articles3.csv', nrows=300000)

all_news = all_news.rename(columns={'title': 'headline'})

all_news['fake'] = 0

data = all_news[['headline', 'fake']]



# data = pd.concat([data, all_news])

data.head()
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



def format_data(data, max_features, maxlen, tokenizer=None, shuffle=False):

    if shuffle:

        data = data.sample(frac=1).reset_index(drop=True)

    

    data['headline'] = data['headline'].apply(lambda x: str(x).lower())



    X = data['headline']

    Y = data['fake'].values # 0: Real; 1: Fake



    if not tokenizer:

        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"

        tokenizer = Tokenizer(num_words=max_features, filters=filters)

        tokenizer.fit_on_texts(list(X))



    X = tokenizer.texts_to_sequences(X)

    X = pad_sequences(X, maxlen=maxlen)



    return X, Y, tokenizer
max_features, max_len = 5000, 25

X, Y, tokenizer = format_data(data, max_features, max_len, shuffle=True)

X_eval, Y_eval, tokenizer = format_data(eval_data, max_features, max_len, tokenizer=tokenizer)
import pickle

pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, LSTM

from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model

from keras import regularizers



epochs=20



# Input shape

inp = Input(shape=(max_len,))



encoder = Embedding(max_features, 50)(inp)

encoder = Bidirectional(LSTM(75, return_sequences=True))(encoder)

encoder = Bidirectional(LSTM(25, return_sequences=True,

                        activity_regularizer=regularizers.l1(10e-5)))(encoder)



decoder = Bidirectional(LSTM(75, return_sequences=True))(encoder)

decoder = GlobalMaxPooling1D()(decoder)

decoder = Dense(50, activation='relu')(decoder)

decoder = Dense(max_len)(decoder)



model = Model(inputs=inp, outputs=decoder)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X, X, epochs=epochs, batch_size=64, verbose=1)



model.save_weights('model{}.h5'.format(epochs))
model.evaluate(X, X)
results = model.predict(X_eval, batch_size=1, verbose=1)
mse = np.mean(np.power(X_eval - results, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                         'true_class': Y_eval})

error_df.describe()
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)
LABELS = ['REAL', 'FAKE']

best, threshold = -1, -1



# General Search

for t in range(0, 3500000, 10000):

    y_pred = [1 if e > t else 0 for e in error_df.reconstruction_error.values]

    score = f1_score(y_pred, error_df.true_class, average='micro', labels=[0, 1])

    if score > best:

        best, threshold = score, t



# Specialized Search around general best

for t in range(threshold-10000, threshold+10000):

    y_pred = [1 if e > t else 0 for e in error_df.reconstruction_error.values]

    score = f1_score(y_pred, error_df.true_class, average='micro', labels=[0, 1])

    if score > best:

        best, threshold = score, t



print(threshold, best)
import matplotlib.pyplot as plt

import seaborn as sns



groups = error_df.groupby('true_class')

fig, ax = plt.subplots()



for name, group in groups:

    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',

            label="Fake" if name == 1 else "Real")



ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.show();
LABELS = ['FAKE', 'REAL']

errors = error_df.reconstruction_error.values

y_pred = [1 if e > threshold else 0 for e in errors] # final predictions

conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
from sklearn.metrics import f1_score



def accuracy_f1(preds, correct):

    return f1_score(preds, correct, average='micro', labels=[0, 1])



accuracy_f1(y_pred, error_df.true_class)
from sklearn.preprocessing import MinMaxScaler

minmax_0_05 = MinMaxScaler(feature_range=(0, 0.5))

minmax_05_1 = MinMaxScaler(feature_range=(0.5, 1))
errors_below = np.array([i for i, e in enumerate(errors) if e <= threshold])

errors_above = np.array([i for i, e in enumerate(errors) if e > threshold])



minmax_0_05.fit(errors[errors_below].reshape(-1, 1))

minmax_05_1.fit(errors[errors_above].reshape(-1, 1))
errors_mm = np.array([minmax_0_05.transform(e.reshape(1, -1)) if i in errors_below

                      else minmax_05_1.transform(e.reshape(1, -1))

                      for i, e in enumerate(errors)]).flatten()



y_pred2 = [1 if e > 0.5 else 0 for e in errors_mm]
def accuracy_percentile(preds, Y_validate):

    real_correct, fake_correct, total_correct = 0, 0, 0

    _, (fake_count, real_count) = np.unique(Y_validate, return_counts=True)



    for i, r in enumerate(preds):

        if r == Y_validate[i]:

            total_correct += 1

            if r == 0:

                fake_correct += 1

            else:

                real_correct += 1



    print('Real Accuracy:', real_correct/real_count * 100, '%')

    print('Fake Accuracy:', fake_correct/fake_count * 100, '%')

    print('Total Accuracy:', total_correct/(real_count + fake_count) * 100, '%')





accuracy_percentile(y_pred2, error_df.true_class)