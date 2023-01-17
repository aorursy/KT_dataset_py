import gc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from xgboost import XGBClassifier





gc.enable()
TestRun = False



test_config = {

    "vocab_size"    :200,

    "embedding_dim" :20,

    "max_length"    :50,

    "max_tfidf_features" :500,

    "max_ngrams" :1,

    "n_epochs": 3}



main_config = {

    "vocab_size"    :6000,

    "embedding_dim" :200,

    "max_length"    :6000,

    "max_tfidf_features" :2000,

    "max_ngrams" :1,

    "n_epochs": 12}



if TestRun:

    config = test_config

else:

    config = main_config

    
train_df = pd.read_csv("/kaggle/input/fake-news/train.csv")

print(train_df.shape)

train_df.head()
test_df = pd.read_csv("/kaggle/input/fake-news/test.csv")

print(test_df.shape)

test_df.head()
train_df["full_text"] = train_df["title"] + " " + train_df["author"] + " " + train_df["text"]

test_df["full_text"] = test_df["title"] + " " + test_df["author"] + " " + test_df["text"]



text_lengths = train_df["full_text"].astype(str).apply(len)

text_lengths.plot(kind="hist", bins=200)

plt.yscale("log")

text_lengths.describe()
# %%time



# transformer = TfidfVectorizer(ngram_range=(1, config["max_ngrams"]), max_features=config["max_tfidf_features"])

# transformer.fit(train_df['full_text'].astype(str).values)

# tfidf = transformer.transform(train_df['full_text'].astype(str).values)

# print(tfidf.shape)



# test_tfidf = transformer.transform(test_df['full_text'].astype(str).values)

# print(test_tfidf.shape)
sentences = train_df["full_text"].astype(str).tolist()

test_sentences = test_df["full_text"].astype(str).tolist()

labels = train_df["label"].values
%%time



tokenizer = Tokenizer(oov_token = "<OOV>", num_words=config["vocab_size"])

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

print(len(word_index))



sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding = 'post', maxlen=config["max_length"])

print(padded[0])

print(padded.shape)



test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, padding = 'post', maxlen=config["max_length"])

print(test_padded[0])

print(test_padded.shape)
# model with multiple inputs: token sequence, tfidf

def get_model():



    keras.backend.clear_session()



    input1 = keras.layers.Input(shape=(config["max_length"],))

    x1 = keras.layers.Embedding(input_dim=config["vocab_size"], output_dim=config["embedding_dim"], input_length=config["max_length"])(input1)

    x1 = keras.layers.Conv1D(128, 3, 2)(x1)

    x1 = keras.layers.MaxPool1D(2)(x1)

    x1 = keras.layers.Conv1D(64, 3, 2)(x1)

    x1 = keras.layers.MaxPool1D(2)(x1)

    x1 = keras.layers.Flatten()(x1)

   

    x = keras.layers.Dropout(0.5)(x1)

    x = keras.layers.Dense(64, activation = 'relu')(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(32, activation = 'relu')(x)

    x = keras.layers.Dense(2, activation = 'softmax')(x)



    model = keras.Model([input1],[x])

    return model



model = get_model()

model.summary()
np.random.seed(1291295)

X_train, X_test, y_train, y_test = train_test_split(padded, keras.utils.to_categorical(labels), test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
callbacks=[

    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, 

                                  verbose=1, mode="min", restore_best_weights=True),

    keras.callbacks.ModelCheckpoint(filepath="best_model.hdf5", verbose=1, save_best_only=True),

    keras.callbacks.ReduceLROnPlateau(factor=0.7, verbose=1, patience=3, min_delta=0.0001)

]



np.random.seed(1291295)

model = get_model()

model.compile(loss=tf.keras.losses.categorical_crossentropy,

                    optimizer=tf.keras.optimizers.Adam(lr=0.01),

                    metrics=[keras.metrics.AUC(), keras.metrics.Accuracy()])



history = model.fit([X_train], y_train, epochs=config["n_epochs"],

                    batch_size=64, validation_data=([X_val], y_val), callbacks=callbacks)
model = keras.models.load_model('best_model.hdf5')
metric_toplot = "loss"

plt.plot(history.epoch, history.history[metric_toplot], ".:", label="loss")

plt.plot(history.epoch, history.history["val_"+metric_toplot], ".:", label="val_loss")

plt.legend()
encoder = keras.models.Sequential(

    model.layers[:-3]

)

encoder.summary()
X_train_enc = encoder.predict([X_train])

X_val_enc = encoder.predict([X_val])

X_test_enc = encoder.predict([X_test])
%%time



xgb = XGBClassifier(n_estimators=1000, max_depth=4, verbosity=1,

                   n_jobs=2, colsample_bytree=0.7, random_state=1291,

                   min_child_weight=5)

xgb.fit(X_train_enc.reshape(len(X_train_enc),-1), y_train.argmax(1))



train_score = xgb.score(X_train_enc.reshape(len(X_train_enc),-1), y_train.argmax(1))

val_score = xgb.score(X_val_enc.reshape(len(X_val_enc),-1), y_val.argmax(1))

test_score = xgb.score(X_test_enc.reshape(len(X_test_enc),-1), y_test.argmax(1))



train_score, val_score, test_score 
y_test_pred = xgb.predict(X_test_enc.reshape(len(X_test_enc),-1))
cr = classification_report(y_test[:,1], y_test_pred)

print(cr)
X_enc = encoder.predict([test_padded])

y_submit_pred = xgb.predict(X_enc.reshape(len(X_enc),-1))

submit_df = pd.DataFrame({"id":test_df["id"],

                         "label":y_submit_pred})

submit_df.head()
submit_df.to_csv("submit.csv", index=False)
!head submit.csv