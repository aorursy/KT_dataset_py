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
train = False
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Model

from keras.layers import Dense, Embedding, Input

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate

from keras.preprocessing import text as keras_text, sequence as keras_seq

from keras.callbacks import EarlyStopping, ModelCheckpoint

import pickle

maxlen = 512
if not train:

    with open('../input/cnn-classification-with-dilations-outputs/tokenizer.pickle', 'rb') as handle:

        tokenizer = pickle.load(handle)
def build_model(conv_layers = 2, 

                dilation_rates = [0, 2, 4, 8, 16], 

                embed_size = 256):

    inp1 = Input(shape=(None, ))

    inp2 = Input(shape=(None, ))

    x1 = Embedding(input_dim = len(tokenizer.word_counts)+1, 

                  output_dim = embed_size)(inp1)

    x2 = Embedding(input_dim = len(tokenizer.word_counts)+1, 

                  output_dim = embed_size)(inp2)

    prefilt_x1 = Dropout(0.25)(x1)

    prefilt_x2 = Dropout(0.25)(x2)

    out_conv = []

    # dilation rate lets us use ngrams and skip grams to process

    count = 0

    for prefilt_x in [prefilt_x1, prefilt_x2]:

        count += 1

        for dilation_rate in dilation_rates:

            x = prefilt_x

            for i in range(2):

                if dilation_rate>0:

                    x = Conv1D(16*2**(i), 

                               kernel_size = 3, 

                               dilation_rate = dilation_rate,

                              activation = 'relu',

                              name = 'ngram_{}_cnn_{}'.format(dilation_rate, str(count)+str(i))

                              )(x)

                else:

                    x = Conv1D(16*2**(i), 

                               kernel_size = 1,

                              activation = 'relu',

                              name = 'word_fcl_{}'.format(str(count)+str(i)))(x)

            out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]

    x = concatenate(out_conv, axis = -1)    

    x = Dense(64, activation='relu')(x)

    x = Dropout(0.1)(x)

    x = Dense(32, activation='relu')(x)

    x = Dropout(0.1)(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inp1, inp2], outputs=x)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model



model = build_model()

model.summary()
test_data = pd.read_csv("../input/quora-question-pairs/test.csv")
if train:

    train_data = pd.read_csv("../input/train.csv")



    from collections import Counter

    Counter(train_data.is_duplicate)



    #train data shape

    train_data.shape



    question1_ids = train_data[["qid1", "question1"]]

    question2_ids = train_data[["qid2", "question2"]]

    question2_ids.rename(columns = {'qid2':'qid1', 'question2':'question1'}, inplace = True)



    question1_ids = question1_ids.drop_duplicates("qid1", keep="last")

    question2_ids = question2_ids.drop_duplicates("qid1", keep="last")

    question_ids = pd.concat([question1_ids, question2_ids])



    import gc

    del question1_ids, question2_ids

    gc.collect()



    question_ids = question_ids.drop_duplicates("qid1", keep="last")

    question_ids["len"] = question_ids.question1.map(str).apply(len)



    question_ids.sort_values(by=["len"]).reset_index(drop=True)



    all_test_question = pd.concat([test_data.question1, test_data.question2])



    all_test_question = all_test_question.drop_duplicates(keep=False)



    all_sentences = list(map(str, pd.concat([all_test_question, question_ids.question1]).tolist()))



    tokenizer = keras_text.Tokenizer(char_level = True)

    tokenizer.fit_on_texts(all_sentences)



    import pickle

    with open('tokenizer.pickle', 'wb') as handle:

        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # train data

    list_tokenized_q1 = tokenizer.texts_to_sequences(train_data.question1.map(str))

    list_tokenized_q2 = tokenizer.texts_to_sequences(train_data.question2.map(str))

    X_t_q1 = keras_seq.pad_sequences(list_tokenized_q1, maxlen=maxlen)

    X_t_q2 = keras_seq.pad_sequences(list_tokenized_q2, maxlen=maxlen)



    y = train_data.is_duplicate



    from sklearn.model_selection import train_test_split

    print('Distribution of Total Positive Labels (important for validation)')

    print(pd.value_counts(y))

    x_indicies = np.array(range(len(X_t_q1)))

    X_train_indicies, X_test_indicies, y_train, y_test = train_test_split(x_indicies, y, 

                                                            test_size = 0.2, 

                                                            stratify = y,

                                                           random_state = 2017)



    batch_size = 256 # large enough that some other labels come in

    epochs = 100



    file_path="best_weights.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=100)



    callbacks_list = [checkpoint, early] #early

    model.fit([X_t_q1[X_train_indicies], X_t_q2[X_train_indicies]], y_train, 

              validation_data=([X_t_q1[X_test_indicies], X_t_q2[X_test_indicies]], y_test),

              batch_size=batch_size, 

              epochs=epochs, 

              shuffle = True,

              callbacks=callbacks_list)



    import gc

    del train_data, X_t_q1, X_t_q2, list_tokenized_q1, list_tokenized_q2, all_sentences

    gc.collect()
# test data

list_tokenized_test_q1 = tokenizer.texts_to_sequences(test_data.question1.map(str))

list_tokenized_test_q2 = tokenizer.texts_to_sequences(test_data.question2.map(str))
file_path="../input/cnn-classification-with-dilations-outputs/best_weights.h5"

model.load_weights(file_path)
end = 100000

prediction_batchs = []

for start in range(0, len(list_tokenized_test_q1), end):

    X_te_q1 = keras_seq.pad_sequences(list_tokenized_test_q1[start: start+end], maxlen=maxlen)

    X_te_q2 = keras_seq.pad_sequences(list_tokenized_test_q2[start: start+end], maxlen=maxlen)

    predict_batch = model.predict([X_te_q1, X_te_q2])

    prediction_batchs.append(predict_batch)
predictions = np.concatenate(np.array(prediction_batchs)).reshape(len(list_tokenized_test_q1))
len(predictions)
sub = pd.read_csv("../input/quora-question-pairs/sample_submission.csv")
sub["is_duplicate"] = predictions
sub.head()
sub.to_csv("submission.csv", index=False)