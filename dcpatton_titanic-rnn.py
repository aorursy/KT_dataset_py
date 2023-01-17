import pandas as pd

import numpy as np

import tensorflow as tf



print(tf.__version__)

# tfds.list_builders()

seed = 51

tf.random.set_seed(seed)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
train.sample(5)
y_train = train['Survived']
# number of missing values

train.isnull().sum(axis=0)
# number of NAN values

train.isna().sum(axis=0)
# fill empty ages with '-1'

train['Age'] = train['Age'].fillna(-1)

test['Age'] = test['Age'].fillna(-1)

    

train.isna().sum(axis=0)
train['text'] = ''
df = pd.DataFrame()

for index, row in train.iterrows():

    if row.Pclass == 1:

        df = df.append({'text':row.text + 'The passenger class is first'}, ignore_index=True)

    if row.Pclass == 2:

        df = df.append({'text':row.text + 'The passenger class is second'}, ignore_index=True)

    if row.Pclass == 3:

        df = df.append({'text':row.text + 'The passenger class is third'}, ignore_index=True)



train.drop(['text', 'Pclass'], axis=1, inplace=True)

train = pd.concat([train, df], axis=1)
df = pd.DataFrame()

for index, row in train.iterrows():

    if row.Sex == 'female':

        df = df.append({'text':row.text + '. The gender is female'}, ignore_index=True)

    if row.Sex == 'male':

        df = df.append({'text':row.text + '. The gender is male'}, ignore_index=True)



train.drop(['text', 'Sex'], axis=1, inplace=True)

train = pd.concat([train, df], axis=1)
df = pd.DataFrame()

for index, row in train.iterrows():

    if row.Age == -1:

        df = df.append({'text':row.text + '. The age is unknown'}, ignore_index=True)

    elif row.Age < 13:

        df = df.append({'text':row.text + '. This age is of a child'}, ignore_index=True)

    elif row.Age > 65:

        df = df.append({'text':row.text + '. This age is of a senior'}, ignore_index=True)

    elif row.Age > 19 and row.Age < 65:

        df = df.append({'text':row.text + '. This age is of an adult'}, ignore_index=True)

    else:

        df = df.append({'text':row.text + '. This age is of a teenager'}, ignore_index=True)



train.drop(['text', 'Age'], axis=1, inplace=True)

train = pd.concat([train, df], axis=1)
df = pd.DataFrame()

for index, row in train.iterrows():

    if row.SibSp == 0:

        df = df.append({'text':row.text + 'The number of relatives is zero'}, ignore_index=True)

    if row.SibSp == 1:

        df = df.append({'text':row.text + 'The number of relatives is one'}, ignore_index=True)

    if row.SibSp == 2:

        df = df.append({'text':row.text + 'The number of relatives is two'}, ignore_index=True)

    if row.SibSp == 3:

        df = df.append({'text':row.text + 'The number of relatives is three'}, ignore_index=True)

    if row.SibSp == 4:

        df = df.append({'text':row.text + 'The number of relatives is four'}, ignore_index=True)

    if row.SibSp == 5:

        df = df.append({'text':row.text + 'The number of relatives is five'}, ignore_index=True)

    if row.SibSp == 6:

        df = df.append({'text':row.text + 'The number of relatives is six'}, ignore_index=True)

    if row.SibSp == 7:

        df = df.append({'text':row.text + 'The number of relatives is seven'}, ignore_index=True)

    if row.SibSp == 8:

        df = df.append({'text':row.text + 'The number of relatives is eight'}, ignore_index=True)

             

train.drop(['text', 'SibSp'], axis=1, inplace=True)

train = pd.concat([train, df], axis=1)
x_text = train['text']
MAX_LEN = 384
tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, num_words=10000)



def prep_data(s):

    

    desc = []



    for index, value in s.iteritems():

        desc.append(str(value))

        

    tokenizer.fit_on_texts(desc)



    tensor1 = tokenizer.texts_to_sequences(desc)

    tensor1 = tf.keras.preprocessing.sequence.pad_sequences(tensor1, maxlen=MAX_LEN, padding='post')

    

    return tensor1
x_train = prep_data(x_text)
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Bidirectional, LSTM, Dropout

from tensorflow.keras.models import Model



input1 = Input(shape=(MAX_LEN,))



x = Embedding(10000, 64)(input1)

x = Bidirectional(LSTM(64,  return_sequences=True, dropout=0.2))(x)

x = Bidirectional(LSTM(32, dropout=0.2))(x)



output = Dense(1, activation='sigmoid')(x)



model = Model(inputs=[input1], outputs=[output])

model.summary()
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



es = EarlyStopping(monitor='val_loss', patience=19, verbose=1, restore_best_weights=True)

rlp = ReduceLROnPlateau(monitor='val_loss', patience=9, verbose=1)



model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=300, verbose=1, batch_size=256

                   , validation_split=0.2, callbacks=[es, rlp]).history
import matplotlib.pyplot as plt

import seaborn as sns



fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Train loss')

ax1.plot(history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['acc'], label='Train accuracy')

ax2.plot(history['val_acc'], label='Validation accuracy')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()
test['text'] = ''
df = pd.DataFrame()

for index, row in test.iterrows():

    if row.Pclass == 1:

        df = df.append({'text':row.text + 'The passenger class is first'}, ignore_index=True)

    if row.Pclass == 2:

        df = df.append({'text':row.text + 'The passenger class is second'}, ignore_index=True)

    if row.Pclass == 3:

        df = df.append({'text':row.text + 'The passenger class is third'}, ignore_index=True)



test.drop(['text', 'Pclass'], axis=1, inplace=True)

test = pd.concat([test, df], axis=1)
df = pd.DataFrame()

for index, row in test.iterrows():

    if row.Sex == 'female':

        df = df.append({'text':row.text + '. The gender is female'}, ignore_index=True)

    if row.Sex == 'male':

        df = df.append({'text':row.text + '. The gender is male'}, ignore_index=True)



test.drop(['text', 'Sex'], axis=1, inplace=True)

test = pd.concat([test, df], axis=1)
df = pd.DataFrame()

for index, row in test.iterrows():

    if row.Age == -1:

        df = df.append({'text':row.text + '. The age is unknown'}, ignore_index=True)

    elif row.Age < 13:

        df = df.append({'text':row.text + '. This age is of a child'}, ignore_index=True)

    elif row.Age > 65:

        df = df.append({'text':row.text + '. This age is of a senior'}, ignore_index=True)

    elif row.Age > 19 and row.Age < 65:

        df = df.append({'text':row.text + '. This age is of an adult'}, ignore_index=True)

    else:

        df = df.append({'text':row.text + '. This age is of a teenager'}, ignore_index=True)



test.drop(['text', 'Age'], axis=1, inplace=True)

test = pd.concat([test, df], axis=1)
df = pd.DataFrame()

for index, row in test.iterrows():

    if row.SibSp == 0:

        df = df.append({'text':row.text + 'The number of relatives is zero'}, ignore_index=True)

    if row.SibSp == 1:

        df = df.append({'text':row.text + 'The number of relatives is one'}, ignore_index=True)

    if row.SibSp == 2:

        df = df.append({'text':row.text + 'The number of relatives is two'}, ignore_index=True)

    if row.SibSp == 3:

        df = df.append({'text':row.text + 'The number of relatives is three'}, ignore_index=True)

    if row.SibSp == 4:

        df = df.append({'text':row.text + 'The number of relatives is four'}, ignore_index=True)

    if row.SibSp == 5:

        df = df.append({'text':row.text + 'The number of relatives is five'}, ignore_index=True)

    if row.SibSp == 6:

        df = df.append({'text':row.text + 'The number of relatives is six'}, ignore_index=True)

    if row.SibSp == 7:

        df = df.append({'text':row.text + 'The number of relatives is seven'}, ignore_index=True)

    if row.SibSp == 8:

        df = df.append({'text':row.text + 'The number of relatives is eight'}, ignore_index=True)

             

test.drop(['text', 'SibSp'], axis=1, inplace=True)

test = pd.concat([test, df], axis=1)
x_test = prep_data(test['text'])
predictions = model.predict(x_test, batch_size=32, verbose=1)

predictions = [0 if pred<0.5 else 1 for pred in predictions ]

# print(predictions)



submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

submission['Survived'] = predictions

submission.to_csv("submission.csv", index=False)