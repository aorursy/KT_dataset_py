import numpy as np

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, Flatten, Dense

from keras.models import Sequential

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
sarcasm_df = pd.read_json("/kaggle/input/Sarcasm_Headlines_Dataset.json", lines=True)
print('json data imported')

print(f'Number of records: {len(sarcasm_df)}')

print(f'Number of columns: {len(sarcasm_df.columns)}')

print(sarcasm_df.head())

print(f'Column names are: {sarcasm_df.columns}')

print(sarcasm_df.iloc[:4])

print(f'Number of records where "is_sarcastic" is 1: {len(sarcasm_df[sarcasm_df["is_sarcastic"] == 1])}')

print(f'Number of records where "is_sarcastic" is 0: {len(sarcasm_df[sarcasm_df["is_sarcastic"] == 0])}')
INPUT_DIM = 50000

MAX_LENGTH = 50

OUTPUT_DIM = 6

EPOCHS = 30

BATCH_SIZE = 512
sarcasm_df = sarcasm_df[['headline','is_sarcastic']]



tokenizer = Tokenizer(num_words=INPUT_DIM)

tokenizer.fit_on_texts(sarcasm_df['headline'])



sequences = tokenizer.texts_to_sequences(sarcasm_df['headline'])



sarcasm_df['length'] = sarcasm_df['headline'].apply(lambda x: len(x.split(' ')))



count = sarcasm_df.is_sarcastic.value_counts()
data = pad_sequences(sequences=sequences,maxlen=MAX_LENGTH)

label = np.asarray(sarcasm_df.is_sarcastic)

print(f'shape of the data: {data.shape} and label: {label.shape}')



train_dfx = data[:2000]

test_dfx = data[-2000:]

train_dfy = label[:2000]

test_dfy = label[-2000:]
model = Sequential()

model.add(Embedding(input_dim=INPUT_DIM,input_length=MAX_LENGTH,output_dim=OUTPUT_DIM))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()



model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc'])



fully_connected = model.fit(x=train_dfx,y=train_dfy,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(test_dfx,test_dfy))
predicted_value = model.predict(test_dfx)

print(predicted_value[0:10])
model.evaluate(x = test_dfx,y = test_dfy)
train_loss, train_acc = fully_connected.history['loss'], fully_connected.history['acc']

val_loss, val_acc = fully_connected.history['val_loss'], fully_connected.history['val_acc']



eps = range(1,len(train_acc)+1)

plt.plot(eps,train_acc,'g',label='Training accuracy')

plt.plot(eps,val_acc,'o',label='Validation accuracy')

plt.legend()

plt.figure()



plt.plot(eps,train_loss,'g',label='training loss')

plt.plot(eps,val_loss,'o',label='val_loss')

plt.legend()

plt.figure()