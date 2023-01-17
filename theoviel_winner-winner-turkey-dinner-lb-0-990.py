import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt



from keras.models import Model

from keras.layers import Dense, Dropout, Bidirectional, CuDNNGRU, Reshape, GlobalMaxPooling1D, GlobalAveragePooling1D, Input, concatenate, BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import accuracy_score
df_train = pd.read_json('../input/train.json')

df_test = pd.read_json('../input/test.json')
print("Number of train sample : ", df_train.shape[0])

print("Number of test sample : ", df_test.shape[0])
df_train.head()
plt.figure(figsize=(12,8))

sns.countplot(df_train['is_turkey'])

plt.show()
df_train.info()
df_test.info()
df_train["length"] = df_train['audio_embedding'].apply(len)
plt.figure(figsize=(12,8))

plt.yscale('log')

sns.countplot("length", hue="is_turkey", data=df_train)

plt.show()
max_len = 10

feature_size = 128
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
X = pad_sequences(df_train['audio_embedding'], maxlen=10, padding='post')

X_test = pad_sequences(df_test['audio_embedding'], maxlen=10, padding='post')
y = df_train['is_turkey'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)



print(f"Training on {X_train.shape[0]} texts")
def build_model():

    inp = Input(shape=(max_len, feature_size))

    x = BatchNormalization()(inp)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)

    concat = concatenate([avg_pool, max_pool])

    concat = Dense(64, activation="relu")(concat)

    concat = Dropout(0.5)(concat)

    output = Dense(1, activation="sigmoid")(concat)

    model = Model(inputs=inp, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    return model
model = build_model()
model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)
epochs = 30
history = model.fit(X_train, y_train, batch_size=256, epochs=epochs, validation_data=[X_val, y_val], callbacks=[reduce_lr], verbose=2)
plt.figure(figsize=(12,8))

sns.lineplot(range(1, epochs+1), history.history['acc'], label='Train Accuracy')

sns.lineplot(range(1, epochs+1), history.history['val_acc'], label='Test Accuracy')

plt.show()
val = model.evaluate(X_val, y_val, verbose=1)

print("Accuracy on validation data : ", val[1])
model_final = build_model()
history_final = model_final.fit(X, y, epochs=30, batch_size=256, verbose=2)
plt.figure(figsize=(10,6))

sns.lineplot(range(1, 31), history_final.history['acc'], label='Train Accuracy')

plt.show()
y_test = model_final.predict(X_test, verbose=1)
submission = pd.DataFrame({'vid_id': df_test['vid_id'].values, 'is_turkey': list(y_test.flatten())})
submission.head()
submission.to_csv("submission.csv", index=False)