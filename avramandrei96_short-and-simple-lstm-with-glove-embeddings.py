import pandas as pd

from nltk.tokenize import TweetTokenizer

from gensim.corpora.dictionary import Dictionary

import numpy as np

import tensorflow as tf

import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
df_train = pd.read_csv("../input/nlp-getting-started/train.csv", encoding='utf8')

df_test = pd.read_csv("../input/nlp-getting-started/test.csv", encoding='utf8')
with open('../input/glove-global-vectors-for-word-representation/glove.twitter.27B.50d.txt', "r") as file:

    dict_w2v = {}



    for line in file:

        tokens = line.split()



        word = tokens[0]

        vector = np.array(tokens[1:], dtype=np.float32)



        if vector.shape[0] == 50:

            dict_w2v[word] = vector
def clean_data(df):

    # remove any html tag

    df["text"] = df["text"].apply(lambda x: re.sub(r'<.*?>', '', x))

    # replace urls with <url> tag

    df["text"] = df["text"].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '<url>', x))

    # replace user names with <user> tag

    df["text"] = df["text"].apply(lambda x: re.sub(r'@[a-zA-Z0-9_]+', '<user>', x))

    # replace hashtags with <hashtag> tag

    df["text"] = df["text"].apply(lambda x: re.sub(r'#[a-zA-Z0-9_]+', '<hashtag>', x))

    

    # replace noisy words - here it can be improved

    df["text"] = df["text"].apply(lambda x: x.replace("\x89", "").replace("hÛ_", "").replace("ÛÓ", ""))

    # replace the happy emojis with <smile> tag

    df["text"] = df["text"].apply(lambda x: re.sub(r'(:|;)-?(\)|D|d)', "<smile>", x))

    # replace the sad emojis with <smile> tag

    df["text"] = df["text"].apply(lambda x: re.sub(r'(:|;)-?\(+', "<sad>", x))

    

    return df



df_train = clean_data(df_train)

df_test = clean_data(df_test)
tokenizer = TweetTokenizer()

tokens_train = [tokenizer.tokenize(tweet) for tweet in df_train["text"]]

tokens_test = [tokenizer.tokenize(tweet) for tweet in df_test["text"]]



vocab = Dictionary(tokens_train + tokens_test)

special_tokens = {'<pad>': 0}

vocab.patch_with_special_tokens(special_tokens)



X_train = [vocab.doc2idx(tokens) for tokens in tokens_train]

y_train = df_train["target"].values

X_test = [vocab.doc2idx(tokens) for tokens in tokens_test]



w2v_train = [[dict_w2v[token] if token in dict_w2v else dict_w2v["<unknown>"] for token in list_tokens] for list_tokens in tokens_train]

w2v_test = [[dict_w2v[token] if token in dict_w2v else dict_w2v["<unknown>"] for token in list_tokens] for list_tokens in tokens_test]
X_train = pad_sequences(X_train)

w2v_train = np.array([w2v_seq + [np.zeros(50)] * (X_train.shape[1] - len(w2v_seq)) for w2v_seq in w2v_train])



X_test = pad_sequences(X_test)

w2v_test = np.array([w2v_seq + [np.zeros(50)] * (X_test.shape[1] - len(w2v_seq)) for w2v_seq in w2v_test])
inputs_tokens = tf.keras.layers.Input(shape=X_train.shape[1], name='inputs_tokens')

inputs_w2v = tf.keras.layers.Input(shape=(X_train.shape[1], 50), name='inputs_w2v')

embeddings = tf.keras.layers.Embedding(len(vocab.token2id) + 1, 25, mask_zero=True)(inputs_tokens)

embeddings = tf.keras.layers.SpatialDropout1D(0.2)(embeddings)

lstms = tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.2)(tf.concat((embeddings, inputs_w2v), axis=2))

outputs = tf.keras.layers.Dense(1, activation='sigmoid', 

                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(lstms)



model = tf.keras.models.Model(inputs=[inputs_tokens, inputs_w2v], outputs=outputs)



optimzer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=optimzer, loss='binary_crossentropy', metrics=['accuracy'])
early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 

                                                 patience=5, 

                                                 restore_best_weights=True)

model.fit([X_train, w2v_train], y_train, batch_size=32, epochs=15, validation_split=0.1, callbacks=[early_stop_cb])
y_pred = model([X_test, w2v_test])

y_pred = [0 if y_pred_val < 0.5 else 1 for y_pred_val in y_pred]



df_pred = pd.DataFrame(df_test["id"])

df_pred["target"] = y_pred

df_pred.to_csv("submission.csv", index=False)