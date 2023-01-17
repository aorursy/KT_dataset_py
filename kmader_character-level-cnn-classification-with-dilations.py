import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Model

from keras.layers import Dense, Embedding, Input

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate

from keras.preprocessing import text as keras_text, sequence as keras_seq

from keras.callbacks import EarlyStopping, ModelCheckpoint
# define network parameters

max_features = 64

maxlen = 512
%%time

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train = train.sample(frac=1)



list_sentences_train = train["comment_text"].fillna("unknown").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_test = test["comment_text"].fillna("unknown").values
tokenizer = keras_text.Tokenizer(char_level = True)

tokenizer.fit_on_texts(list(list_sentences_train))

# train data

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

X_t = keras_seq.pad_sequences(list_tokenized_train, maxlen=maxlen)

# test data

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_te = keras_seq.pad_sequences(list_tokenized_test, maxlen=maxlen)
def build_model(conv_layers = 2, 

                dilation_rates = [0, 2, 4, 8, 16], 

                embed_size = 256):

    inp = Input(shape=(None, ))

    x = Embedding(input_dim = len(tokenizer.word_counts)+1, 

                  output_dim = embed_size)(inp)

    prefilt_x = Dropout(0.25)(x)

    out_conv = []

    # dilation rate lets us use ngrams and skip grams to process 

    for dilation_rate in dilation_rates:

        x = prefilt_x

        for i in range(2):

            if dilation_rate>0:

                x = Conv1D(16*2**(i), 

                           kernel_size = 3, 

                           dilation_rate = dilation_rate,

                          activation = 'relu',

                          name = 'ngram_{}_cnn_{}'.format(dilation_rate, i)

                          )(x)

            else:

                x = Conv1D(16*2**(i), 

                           kernel_size = 1,

                          activation = 'relu',

                          name = 'word_fcl_{}'.format(i))(x)

        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]

    x = concatenate(out_conv, axis = -1)    

    x = Dense(64, activation='relu')(x)

    x = Dropout(0.1)(x)

    x = Dense(32, activation='relu')(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model



model = build_model()

model.summary()
from sklearn.model_selection import train_test_split

any_category_positive = np.sum(y,1)

print('Distribution of Total Positive Labels (important for validation)')

print(pd.value_counts(any_category_positive))

X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y, 

                                                        test_size = 0.2, 

                                                        stratify = any_category_positive,

                                                       random_state = 2017)

print('Training:', X_t_train.shape)

print('Testing:', X_t_test.shape)
batch_size = 128 # large enough that some other labels come in

epochs = 1



file_path="best_weights.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)



callbacks_list = [checkpoint, early] #early

model.fit(X_t_train, y_train, 

          validation_data=(X_t_test, y_test),

          batch_size=batch_size, 

          epochs=epochs, 

          shuffle = True,

          callbacks=callbacks_list)
model.load_weights(file_path)

y_test = model.predict(X_te)

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[list_classes] = y_test

sample_submission.to_csv("predictions.csv", index=False)