import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras import Input, Model
from keras.layers import Dense, CuDNNGRU, GlobalAveragePooling1D, Embedding, Dropout
from keras.layers import Bidirectional

from gensim.models import KeyedVectors
config = {
    "embeddings": {
        "word2vec": {
            "filepath": "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
        }, 
        "fasttext": {
            "filepath": "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec",
        },
        "glove": {
            "filepath": "../input/embeddings/glove.840B.300d/glove.840B.300d.txt",
        },
        "paragram": {
            "filepath": "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
        }
    }
}
IS_KAGGLE_KERNEL = 'KAGGLE_WORKING_DIR' in os.environ
EXAM_NAME = "exam-w2v-gru"
SEQUENCE_MAXLEN = 100
MAX_FEATURES = 50000
EMBEDDING_SIZE = 300
EMBEDDING_NAME = "word2vec"
if IS_KAGGLE_KERNEL:
    SUBMISSION_FILEPATH = "submission.csv"
else:
    SUBMISSION_FILEPATH = "../output/submission_%s.csv" % EXAM_NAME
print("Submission file path: %s" % SUBMISSION_FILEPATH)
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=777)
y_train = train_df["target"].values
y_valid = valid_df["target"].values
def transform(df):
    return df["question_text"].fillna("__NA__").values
train_texts = transform(train_df)
valid_texts = transform(valid_df)
test_texts = transform(test_df)
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(np.concatenate([train_texts, valid_texts, test_texts]))
train_sequences = tokenizer.texts_to_sequences(train_texts)
valid_sequences = tokenizer.texts_to_sequences(valid_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
X_train = sequence.pad_sequences(train_sequences, maxlen=SEQUENCE_MAXLEN)
X_valid = sequence.pad_sequences(valid_sequences, maxlen=SEQUENCE_MAXLEN)
X_test = sequence.pad_sequences(test_sequences, maxlen=SEQUENCE_MAXLEN)
def make_embedding_matrix(filepath, embedding_name):
    if embedding_name == "word2vec":
        embedding_index = KeyedVectors.load_word2vec_format(filepath, binary=True)
        A = embedding_index.wv.vectors
    else:
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype=np.float32)
        embedding_index = dict(get_coefs(*_.split(" ")) for _ in open(filepath, encoding="utf8", errors="ignore") if len(_) > 100)
        A = np.stack(embedding_index.values())
    avg = np.mean(A)
    std = np.std(A)
    num_words = min(MAX_FEATURES, len(tokenizer.word_index))
    embedding_matrix = np.random.normal(avg, std, (num_words, EMBEDDING_SIZE))
    for word, index in tokenizer.word_index.items():
        if index >= MAX_FEATURES:
            continue
        if word in embedding_index:
            embedding_matrix[index] = embedding_index[word]
    return embedding_matrix
def build_model(embedding_name):
    embedding_matrix = make_embedding_matrix(
        filepath = config["embeddings"][embedding_name]["filepath"],
        embedding_name = embedding_name
    )
    x_in = Input(shape=(SEQUENCE_MAXLEN, ))
    x = Embedding(MAX_FEATURES, EMBEDDING_SIZE, weights=[embedding_matrix])(x_in)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    prediction = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[x_in], outputs=[prediction])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
model = build_model(EMBEDDING_NAME)
model.fit(
    X_train,
    y_train, 
    batch_size=512,
    epochs=2,
    validation_data=(X_valid, y_valid)
);
y_preda_valid = model.predict(X_valid, batch_size=1024)
best_threshold = 0.5
best_score = -1
for threshold in np.arange(0.2, 0.8, 0.01):
    y_pred_valid = (y_preda_valid > threshold).astype(int)
    score = f1_score(y_valid, y_pred_valid)
    if best_score < score:
        best_score = score
        best_threshold = threshold
        print("F1=%.4f, th=%.4f" % (score, threshold))
y_preda = model.predict(X_test, batch_size=1024)
y_pred = (y_preda.flatten() > best_threshold).astype(int)
test_df["prediction"] = y_pred
test_df[["qid", "prediction"]].to_csv(SUBMISSION_FILEPATH, index=False)
