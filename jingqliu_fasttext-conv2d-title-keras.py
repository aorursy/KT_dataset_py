import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
train = pd.read_csv('../input/avito-demand-prediction/train.csv')
test = pd.read_csv('../input/avito-demand-prediction/test.csv')
submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
X_train = train["title"].fillna("fillna").values
y_train = train['deal_probability'].values
X_test = test["title"].fillna("fillna").values
max_features = 100000
maxlen = 15
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('../input/fasttext-russian-2m/wiki.ru.vec'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

filter_sizes = [1,2,3,4]
num_filters = 32
inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = SpatialDropout1D(0.3)(x)
x = Reshape((maxlen, embed_size, 1))(x)
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal', activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal', activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal', activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal', activation='elu')(x)
maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)    
z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
z = Flatten()(z)
z = Dropout(0.1)(z)
outp = Dense(2, activation="softmax")(z)
model = Model(inputs=inp, outputs=outp)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
batch_size = 256
epochs = 3
y_train = np.array(pd.concat([pd.DataFrame(y_train),pd.DataFrame(1-y_train)],axis=1))
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
y_pred = model.predict(x_test, batch_size=1024)
pd.DataFrame(y_pred).to_csv('df_test_title_formal.csv',index=False)
submission['deal_probability'] = y_pred[:,0]
submission.to_csv('submission.csv', index=False)