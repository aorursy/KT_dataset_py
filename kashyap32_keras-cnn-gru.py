# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

from keras.models import Model

from keras.layers import Dense, Embedding, Input,GRU

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate

from keras.preprocessing import text, sequence



from keras.callbacks import EarlyStopping, ModelCheckpoint
max_features = 20000

maxlen = 100
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.head(10))

list_sentences_train = train["comment_text"].fillna("unknown").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_test = test["comment_text"].fillna("unknown").values
print(list_sentences_train[0])

y[0]
tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

# train data

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

# test data

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
def cnn_rnn():

    embed_size = 256

    inp = Input(shape=(maxlen, ))

    main = Embedding(max_features, embed_size)(inp)

    main = Dropout(0.2)(main)

    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)

    main = MaxPooling1D(pool_size=2)(main)

    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)

    main = MaxPooling1D(pool_size=2)(main)

    main = GRU(32)(main)

    main = Dense(16, activation="relu")(main)

    main = Dense(6, activation="sigmoid")(main)

    model = Model(inputs=inp, outputs=main)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()     

    return model
model = cnn_rnn()

model.summary()
from sklearn.model_selection import train_test_split



print('Positive Labels ')

any_category_positive = np.sum(y,1)

print(pd.value_counts(any_category_positive))

X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y, 

                                                        test_size = 0.10, 

                                                        )

print('Training:', X_t_train.shape)

print('Testing:', X_t_test.shape)
batch_size = 128 

epochs = 3



file_path="model_best.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)



callbacks_list = [checkpoint, early] #early

model.fit(X_t_train, y_train, 

          validation_data=(X_t_test, y_test),

          batch_size=batch_size, 

          epochs=epochs, 

          shuffle = True,

          callbacks=callbacks_list)

model.save('Whole_model.h5')
model.load_weights(file_path)

y_test = model.predict(X_te)

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[list_classes] = y_test

sample_submission.to_csv("predictions.csv", index=False)
#For fun not for testing purpose!!

test=["This is your last warning. You will be blocked from editing the next time you vandalize a page, as you did with this edit to Geb.  |Parlez ici "]



tokenizer.fit_on_texts(list(test))

# train data

test_token = tokenizer.texts_to_sequences(test)

test_2 = sequence.pad_sequences(test_token, maxlen=maxlen)
np.argmax(model.predict(test_2))
model.predict(test_2)
pred=pd.read_csv('predictions.csv')

pred.head()