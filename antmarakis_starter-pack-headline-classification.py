import pandas as pd



d_fake = pd.read_csv('../input/fnn_politics_fake.csv')

headlines_fake = d_fake.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})

headlines_fake['fake'] = 1



d_real = pd.read_csv('../input/fnn_politics_real.csv')

headlines_real = d_real.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})

headlines_real['fake'] = 0



data = pd.concat([headlines_fake, headlines_real])
data = data.sample(frac=1).reset_index(drop=True)

data.head()
data['fake'].value_counts()
import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



def format_data(data, max_features, maxlen):

    data = data.sample(frac=1).reset_index(drop=True)

    data['headline'] = data['headline'].apply(lambda x: x.lower())



    Y = data['fake'].values # 0: Real; 1: Fake

    X = data['headline']



    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(X))



    X = tokenizer.texts_to_sequences(X)

    X = pad_sequences(X, maxlen=maxlen)



    return X, Y
max_features, max_len = 3500, 25

X, Y = format_data(data, max_features, max_len)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

len(X_train), len(X_test)
from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout

from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model



# Input shape

inp = Input(shape=(max_len,))



# Embedding and GRU

x = Embedding(max_features, 300)(inp)

x = SpatialDropout1D(0.33)(x)

x = Bidirectional(GRU(50, return_sequences=True))(x)



# Pooling

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([avg_pool, max_pool])



# Output layer

output = Dense(1, activation='sigmoid')(conc)



model = Model(inputs=inp, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# model.load_weights('Weights/gru5.h5')

model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)
results = model.predict(X_test, batch_size=1, verbose=1)
def convert_to_preds(results):

    """Converts probabilistic results in [0, 1] to

    binary values, 0 and 1."""

    return [1 if r > 0.5 else 0 for r in results]



preds = convert_to_preds(results)
def accuracy_percentile(preds, Y_validate):

    """Return the percentage of correct predictions for each class and in total"""

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



accuracy_percentile(preds, Y_test)
from sklearn.metrics import f1_score



def accuracy_f1(preds, correct):

    """Returns F1-Score for predictions"""

    return f1_score(preds, correct, average='micro', labels=[0, 1])



accuracy_f1(preds, Y_test)