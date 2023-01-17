from keras import layers, models, optimizers, datasets, preprocessing

import os

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.utils import np_utils



base_dir='../input'

train_dir=os.path.join(base_dir, '')

validation_dir=os.path.join(base_dir, '')



# paramater

batch_size=128

seed=1470



def load_data():

    data = pd.read_csv(os.path.join(train_dir, 'train.tsv'), sep='\t')

    x_train = [line['Phrase'] for i, line in data.iterrows()]

    y_train = [line['Sentiment'] for i, line in data.iterrows()]

    return x_train, y_train



def get_model():

    main_input = layers.Input(shape=(max_len,))

    embedded = layers.Embedding(max_features, 32)(main_input)

    rnn_output = layers.GRU(32)(embedded)

    dense1 = layers.Dense(32, activation='relu')(rnn_output)

    dense1 = layers.Dropout(0.3)(dense1)

    dense2 = layers.Dense(5, activation='softmax')(dense1)

    model = models.Model(inputs=main_input, outputs=dense2)

    model.compile(optimizer='rmsprop',

                  loss='categorical_crossentropy',

                  metrics=['acc'])

    model.summary()

    return model





max_features = 10000

max_len = 200



x_train, y_train = load_data()

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)

y_train = np_utils.to_categorical(y_train)

model = get_model()

history = model.fit(x_train, y_train,

                    epochs=10,

                    batch_size=128,

                    )



# predict

test_data = pd.read_csv('../input/test.tsv', sep='\t')

x_test = [line['Phrase'] for i, line in test_data.iterrows()]

x_test = tokenizer.texts_to_sequences(x_test)

x_test= preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

test_pred = model.predict(x_test)

test_pred = test_pred.argmax(axis=1)

pred_df = pd.DataFrame({'PhraseId':test_data['PhraseId'].values, 'Sentiment': test_pred})

pred_df.to_csv('submit.csv', index=False)




