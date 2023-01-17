import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Concatenate, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau



from tensorflow.keras.utils import plot_model

from tensorflow.keras.regularizers import l1



from gensim.models import KeyedVectors, Word2Vec

import multiprocessing



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import pandas as pd

import numpy as np
with open("../input/isods-dataset/w2v_train.txt", "r") as f:

    sentences = f.read().splitlines()

    

sentences = [s.split() for s in sentences if s]
cores = multiprocessing.cpu_count()
sentences[-1]
%%time

w2v_model = Word2Vec(sentences,

                     window=2,

                     size=300,

                     min_count=5,

                     sg=1,

                     hs=1,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=cores-1)
len(w2v_model.wv.vocab)
w2v_model.wv.most_similar(positive=["ctkm"])
w2v_model.save("w2v.model")
# max len cho keras tokenizer

padded_max_len = 500



# Callback tính roc sau mỗi epoch

class RocAucCallback(Callback):

    def __init__(self, X_train, y_train, X_val, y_val):

        self.x = X_train

        self.y = y_train

        self.x_val = X_val

        self.y_val = y_val



    def on_epoch_end(self, epoch, logs={}):

        y_pred_train = self.model.predict(self.x)

        roc_train = roc_auc_score(self.y, y_pred_train)

        y_pred_val = self.model.predict(self.x_val)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print(f"\n### roc-auc_train: {round(roc_train, 4)} - roc-auc_val: {round(roc_val, 4)} ###\n")
df = pd.read_csv("../input/isods-dataset/preprocessed_df.csv")

df = df.fillna("")

y = np.array([0 if x == "N" else 1 for x in df.is_unsatisfied])

df.tail()
train_df, test_df = train_test_split(df, test_size=0.3, stratify=y, random_state=42)
import seaborn as sns

import matplotlib.pyplot as plt



fig, ax = plt.subplots(1,2)

sns.countplot(train_df.is_unsatisfied, ax=ax[0])

sns.countplot(test_df.is_unsatisfied, ax=ax[1])

fig.show()
tokenizer = Tokenizer()

tokenizer.fit_on_texts(df.question.to_list() + df.answer.to_list())
train_ques = tokenizer.texts_to_sequences(train_df.question.to_list())

train_ans = tokenizer.texts_to_sequences(train_df.answer.to_list())
test_ques = tokenizer.texts_to_sequences(test_df.question.to_list())

test_ans = tokenizer.texts_to_sequences(test_df.answer.to_list())
padded_tr_ques = pad_sequences(train_ques, padding="post", maxlen=padded_max_len)

padded_tr_ans = pad_sequences(train_ans, padding="post", maxlen=padded_max_len)
padded_te_ques = pad_sequences(test_ques, padding="post", maxlen=padded_max_len)

padded_te_ans = pad_sequences(test_ans, padding="post", maxlen=padded_max_len)
y_train = np.array([0 if x == "N" else 1 for x in train_df.is_unsatisfied.values])

y_test =  np.array([0 if x == "N" else 1 for x in test_df.is_unsatisfied.values])
embedding_size = w2v_model.vector_size
embeddings_index = dict()

for key in w2v_model.wv.vocab:

    embeddings_index[key] = w2v_model.wv[key]

print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_size))

for word, i in tokenizer.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
print(f"w2v chỉ chứa {int(np.count_nonzero(embedding_matrix) / embedding_size)} vocab trong {len(tokenizer.word_index)} vocab của cuộc thi!")
import gc

del w2v_model

del embeddings_index

gc.collect()
#reg = l1(1e-1)
in_ques = Input(shape=(padded_max_len, ), name="tokenized_question_input_layer")

in_ans = Input(shape=(padded_max_len, ), name="tokenized_answer_input_layer")
em_ques = Embedding(len(tokenizer.word_index) + 1, 

                    embedding_size, 

                    input_length=padded_max_len, 

                    name="question_embedding", 

                    mask_zero=True, 

                    weights=[embedding_matrix], 

                    trainable=False)(in_ques)



em_ans = Embedding(len(tokenizer.word_index) + 1, 

                   embedding_size, 

                   input_length=padded_max_len, 

                   name="answer_embedding", 

                   mask_zero=True, 

                   weights=[embedding_matrix], 

                   trainable=False)(in_ans)
lstm_ques = LSTM(embedding_size, name="question_lstm")(em_ques)
lstm_ans = LSTM(embedding_size, name="answer_lstm")(em_ans)
concat = Concatenate(axis=1, name="concatenate_layer")([lstm_ques, lstm_ans])
# hidden = Dense(128, activation="relu", name="dense_128")(concat)
# dropout = Dropout(0.3)(hidden)
out = Dense(1, activation="sigmoid", name="sigmoid_output")(concat)
model = Model(inputs=[in_ques, in_ans], outputs=out)



# loss

l = tf.keras.losses.BinaryCrossentropy()



# optimizer

op = tf.keras.optimizers.Adam(learning_rate=1e-1)



# metrics

auc = tf.keras.metrics.AUC()

bin_acc = tf.keras.metrics.BinaryAccuracy()



model.compile(loss=l,

              optimizer=op,

              metrics=[auc, bin_acc])
model.summary()
plot_model(model)
# callbacks

roc_auc = RocAucCallback(X_train=[padded_tr_ques, padded_tr_ans], 

                         y_train=y_train, 

                         X_val=[padded_te_ques, padded_te_ans], 

                         y_val=y_test)



es = EarlyStopping(monitor="val_loss", 

                   patience=4, 

                   verbose=1, 

                   restore_best_weights=True)



lrplat = ReduceLROnPlateau(monitor="val_loss", 

                           factor=0.2,

                           patience=2, 

                           verbose=1,

                           min_lr=0.0001)



# fit

model.fit([padded_tr_ques, padded_tr_ans],

          y_train,

          epochs=10,

          batch_size=100,

          validation_data=([padded_te_ques, padded_te_ans], y_test),

          callbacks=[roc_auc, lrplat, es])
model.save("funclstm_w2v")