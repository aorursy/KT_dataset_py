import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
data = pd.read_sql("""
    select
        content,
        case when rating < 3 then 0
            when rating > 4 then 2
            else 1
        end as sentiment
    from reviews
""", sqlite3.connect("/kaggle/input/podcastreviews/database.sqlite"))
data
data.sentiment.value_counts()
class_weights = {k: (len(data) / v) for k, v in data.sentiment.value_counts().items()}
class_weights
%%time
VOCAB_SIZE = 100_000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(data.content)
tokenizer.texts_to_sequences(["This is how we do it. It's Friday night, and I feel all right, and the party's here on the west side."])
SEQ_LEN = 128

def prepare_texts(texts: list) -> np.array:
    return tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=SEQ_LEN)
%%time

x = prepare_texts(data.content)
y = data.sentiment
print(f"Corpus has {np.count_nonzero(x)} words in it")
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=0)
def build_model(vocab_size: int, embed_dim: int, seq_len: int, rnn_dim: int, num_classes: int) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_len))
    model.add(tf.keras.layers.SpatialDropout1D(0.5))
    model.add(tf.keras.layers.LSTM(rnn_dim, return_sequences=True, dropout=0.5))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(rnn_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model
model = build_model(VOCAB_SIZE, 100, SEQ_LEN, 256, 3)
model.summary()
model.fit(x_train, y_train, batch_size=1024, class_weight=class_weights, epochs=5, validation_data=(x_val, y_val))
model.predict(prepare_texts([
    "I love this podcast!",
    "It's not qutie as good as when it started, but I still listen every week.",
    "Not worth listening to.",
    "I used to love this podcast, but now I can't stand the back and forth.",
])).argmax(axis=1)
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(x_val, batch_size=2048)
print(classification_report(y_val, y_pred.argmax(1), target_names=["Negative", "Neutral", "Positive"]))
print(confusion_matrix(y_val, y_pred.argmax(1)))
