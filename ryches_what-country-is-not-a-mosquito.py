# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
X_train = train_df["question_text"].fillna("Is it unethical to sabotage a public kernel?").values
test_df = pd.read_csv("../input/test.csv")
X_test = test_df["question_text"].fillna("Is it unethical to sabotage a public kernel?").values
y = train_df["target"]

train_df[train_df["target"] != 0][["question_text", "target"]]

from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
maxlen = 10
max_features = 30000

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
word_index = tokenizer.word_index
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

train_df["seq"] = sequence.pad_sequences(X_train, maxlen=maxlen).tolist()
test_df["seq"] = sequence.pad_sequences(X_test, maxlen=maxlen).tolist()
import os
import zipfile
import numpy as np
embeddings_index = {}
GLOVE_DIR = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
print(os.listdir("../input/embeddings/glove.840B.300d"))

f = open(GLOVE_DIR)
for line in f:
    values = line.split()
    word = values[0]
    #print(values)
    try:
        coefs = np.asarray(values[1:], dtype='float32')
    except:
        pass
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))





embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)
from keras.layers import concatenate, Flatten, Lambda, Permute, Reshape, merge
def get_model():
    inp = Input(shape=(maxlen, ))
    x = embedding_layer(inp)
    x = Bidirectional(CuDNNGRU(10, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(10, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([
                        avg_pool, 
                        max_pool])

    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=[inp], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()
batch_size = 512
epochs = 8
from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(train_df, y, test_size = 0.05, random_state=42)
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model.compile(loss='binary_crossentropy',
          optimizer= "adam",
          metrics=["acc", f1])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
checkpoint = ModelCheckpoint('gru.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001)
early = EarlyStopping(monitor='val_loss', 
                      mode="min", 
                      patience=10)
callbacks_list = [checkpoint, early, reduceLROnPlat]
#with self train
from keras.models import load_model

hist = model.fit([np.array(X_tra["seq"].tolist())], y_tra, batch_size=batch_size, epochs=epochs,
                  validation_data=([np.array(X_val["seq"].tolist())], y_val),
                  verbose=True, callbacks = callbacks_list)



model = load_model('gru.h5', custom_objects={'f1': f1})

val_pred1 = model.predict([np.array(X_val["seq"].tolist())], batch_size=128)
positives = y_val[y_val > 0]
positive_scores = val_pred1[:, 0][y_val > 0]
pos_text = X_val.loc[positives.index.values]
pos_text
pd.DataFrame(val_pred1).describe()
pred_sort = val_pred1[:, 0][y_val > 0].argsort()[:250][::-1]
pd.DataFrame(val_pred1[y_val > 0][pred_sort]).describe()
X_val[y_val > 0].iloc[pred_sort]

