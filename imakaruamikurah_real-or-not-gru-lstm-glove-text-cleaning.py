import numpy as np
import pandas as pd
import re
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers as ly
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_train = df_train.drop(["id"], axis=1)
df_train = df_train.drop_duplicates()
df_train.head()
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_test = df_test.drop(["id"], axis=1)
df_test.head()
import random

random.seed(42)
sentences = np.array(df_train["text"])
test_sentences = np.array(df_test["text"])

labels = np.array(df_train["target"])

random.shuffle(labels)
random.shuffle(sentences)
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()
def lemmatize_sentence(text):
    words = nltk.word_tokenize(text)
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in words])
    return lemmatized

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)
    return text


def clean_sentence(text):
    text = re.sub(r"http\S+", "", text) # remove urls
    text = re.sub(r'@[^\s]+','',text) # remove usernames
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace("#", "")
    text = text.replace("can't", "can not").replace("won't", "will not").replace("n't", " not")
    text = text.replace("'m", " am").replace("'re", " are").replace("'s", "  is").replace("'d", " would")
    text = text.replace("'ll", " will").replace("'t", " not").replace("'ve", "  have")
    text = remove_emoji(text)
    for word in stop_words:
        text = text.replace(" "+word+" ", " ")
    text = ''.join([i for i in text if not i.isdigit()])
    text = ' '.join([i for i in text.split(' ') if len(i) > 2])
    text = lemmatize_sentence(text.lower())
    return text
sentences = [clean_sentence(sentence) for sentence in sentences]

test_sentences = [clean_sentence(text) for text in test_sentences]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
VOCAB_SIZE = len(tokenizer.word_index)
MAX_LENGTH = 20
EMBEDDING_DIM = 16
SPLIT = 1000
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post", maxlen=MAX_LENGTH, truncating="post")

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding="post",maxlen=MAX_LENGTH, truncating="post")
sentences[:2]
sequences[:2]
padded[:2]
from tensorflow.keras.callbacks import ModelCheckpoint

callback = ModelCheckpoint("model_NLP.h5", monitor="val_accuracy", save_best_only=True)
model = Sequential([   
    ly.Embedding(VOCAB_SIZE+1, EMBEDDING_DIM, input_length=MAX_LENGTH, trainable=False),
    ly.Dropout(0.2),
    ly.Conv1D(64, 5, activation='relu'),
    ly.MaxPooling1D(4),
    ly.LSTM(64),
    ly.Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
model.summary()
train_x = padded[SPLIT:]
train_y = labels[SPLIT:]

val_x = padded[:SPLIT]
val_y = labels[:SPLIT]
history = model.fit(train_x, train_y,
                    epochs=50,
                    validation_data=(val_x, val_y),
                    callbacks=[callback])
def plot_results(his):
    loss = his.history["loss"]
    val_loss = his.history["val_loss"]
    acc = his.history["accuracy"]
    val_acc = his.history["val_accuracy"]

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(["loss", "val_loss"])
    plt.show()

    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(["acc", "val_acc"])
    plt.show()
plot_results(history)
def make_predictions(best_model=True):
    if best_model:
        m = tf.keras.models.load_model("model_NLP.h5")
    else:
        m = model
    
    prediction = m.predict(test_padded)

    for i in range(15):
        result = "REAL" if prediction[i] > 0.5 else "FAKE"
        print(df_test.iloc[i]["text"], " - ", result)
    return prediction
p = make_predictions()
p = make_predictions(False)
embedding_dict = dict()
with open("../input/glove6b200d/glove.6B.200d.txt", "r") as f:
    for line in f:
        vals = line.split()
        word = vals[0]
        vects = np.array(vals[1:], dtype="float32")
        embedding_dict[word] = vects
word_index = tokenizer.word_index

embedding_matrix = np.zeros((VOCAB_SIZE+1, 200))
for word, i in word_index.items():
    if i > VOCAB_SIZE+1:
        continue
    emb_vec = embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i] = emb_vec
model = Sequential([
    ly.Embedding(VOCAB_SIZE+1, 200, embeddings_initializer=Constant(embedding_matrix), input_length=MAX_LENGTH, trainable=False),
    ly.Dropout(0.2),
    ly.Conv1D(64, 5, activation='relu'),
    ly.MaxPooling1D(4),
    ly.LSTM(64),
    ly.Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
model.summary()
history = model.fit(train_x, train_y,
                    epochs=50,
                    validation_data=(val_x, val_y),
                    callbacks=[callback])
plot_results(history)
pred = make_predictions()
pred = make_predictions(False)
def save_results(prediction):
    TEST_RESULTS = []

    for i in range(len(prediction)):
        r = 1 if prediction[i] > 0.5 else 0
        TEST_RESULTS.append(r)
    
    sub_df = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
    sub_df["target"] = TEST_RESULTS
    sub_df.to_csv("submission.csv",index=False)
save_results(pred)
