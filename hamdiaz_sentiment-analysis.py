import numpy as np

import pandas as pd



import os

import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
def viz_report(history, n_epoch):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs_range = range(n_epoch)



    plt.figure(figsize=(9, 5))

    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')

    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')

    plt.title('Training and Validation Accuracy')



    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.title('Training and Validation Loss')

    plt.show()
from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
trainlama = pd.read_csv('/kaggle/input/datasetreview1/train.csv')

testlama = pd.read_csv('/kaggle/input/datasetreview1/test.csv')



proportions = (trainlama['rating'].value_counts()/len(trainlama)).sort_index()



preds = [1]* int(round(proportions[1]*len(testlama)))

preds += [2]* int(round(proportions[2]*len(testlama)))

preds += [3]* int(round(proportions[3]*len(testlama)))

preds += [4]* int(round(proportions[4]*len(testlama)))

preds += [5]* int(round(proportions[5]*len(testlama)))

testlama['rating'] = preds



train = pd.concat([trainlama, testlama], ignore_index=True)

test = pd.read_csv('/kaggle/input/datasetreview2/test.csv')



del trainlama

del testlama



train.shape, test.shape
import re

def clean1(text):

    return re.sub(r'(.)\1+', r'\1\1', text)

    

def clean2(text):

    def convert(c):

        if c in ['!', '?']:

            return ' '+c

        return c

    return ''.join(convert(c.lower()) for c in text if not (c.isdigit() or c in ['.', ',', ':', "'", '"']))
!pip install nltk
from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



wordnet_lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))



def clean3(text):

    return ' '.join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])
train['review'] = train['review'].apply(lambda x: clean1(x))

test['review'] = test['review'].apply(lambda x: clean1(x))



train['review'] = train['review'].apply(lambda x: clean2(x))

test['review'] = test['review'].apply(lambda x: clean2(x))



train['review'] = train['review'].apply(lambda x: clean3(x))

test['review'] = test['review'].apply(lambda x: clean3(x))
train.head()
test.head()
train['word_count'] = train['review'].apply(lambda x: len(x.split()))



plt.figure(figsize=(20,5))

sns.countplot(train['word_count'])
np.percentile(train['word_count'], 50), np.percentile(train['word_count'], 95), train['word_count'].max()
# def distPlot(data, col, hue=None, params={}):

#     plt.figure(figsize=(20,5))

#     if hue:

#         labels = data[hue].unique()

#         for label in labels:

#             sns.distplot(data[data[hue]==label][col], **params);

#         plt.legend(labels=labels)

#     else:

#         sns.distplot(data[col], **params);

#     plt.show()



# distPlot(train, 'word_count', hue='rating', params={'kde': False})
class MultiHeadSelfAttention(layers.Layer):

    def __init__(self, embed_dim, num_heads=8):

        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        if embed_dim % num_heads != 0:

            raise ValueError(

                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"

            )

        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)

        self.key_dense = layers.Dense(embed_dim)

        self.value_dense = layers.Dense(embed_dim)

        self.combine_heads = layers.Dense(embed_dim)



    def attention(self, query, key, value):

        score = tf.matmul(query, key, transpose_b=True)

        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)

        scaled_score = score / tf.math.sqrt(dim_key)

        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)

        return output, weights



    def separate_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])



    def call(self, inputs):

        # x.shape = [batch_size, seq_len, embedding_dim]

        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)

        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)

        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(

            query, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        key = self.separate_heads(

            key, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        value = self.separate_heads(

            value, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(

            attention, perm=[0, 2, 1, 3]

        )  # (batch_size, seq_len, num_heads, projection_dim)

        concat_attention = tf.reshape(

            attention, (batch_size, -1, self.embed_dim)

        )  # (batch_size, seq_len, embed_dim)

        output = self.combine_heads(

            concat_attention

        )  # (batch_size, seq_len, embed_dim)

        return output
class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

        super(TransformerBlock, self).__init__()

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        self.ffn = keras.Sequential(

            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]

        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)

        self.dropout2 = layers.Dropout(rate)



    def call(self, inputs, training):

        attn_output = self.att(inputs)

        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)
class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):

        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)



    def call(self, x):

        maxlen = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=maxlen, delta=1)

        positions = self.pos_emb(positions)

        x = self.token_emb(x)

        return x + positions
top_words = 30000

X_train, X_val, y_train, y_val = train_test_split(train['review'],

                                                  train['rating'],

                                                  test_size=0.2,

                                                  random_state=1,

                                                  stratify=train['rating'])



tk = Tokenizer(num_words=top_words, split=" ")

tk.fit_on_texts(X_train)

X_train = tk.texts_to_sequences(X_train)

X_val = tk.texts_to_sequences(X_val)

X_test = tk.texts_to_sequences(test['review'])



max_review_length = 50

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_val = sequence.pad_sequences(X_val, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
vocab_size = top_words

maxlen = max_review_length
X_train.shape, X_val.shape, X_test.shape
l = LabelBinarizer().fit(y_train.values)

y_train = l.transform(y_train.values)

y_val = l.transform(y_val.values)

l.classes_
embed_dim = 192  # Embedding size for each token

num_heads = 6  # Number of attention heads

ff_dim = 768  # Hidden layer size in feed forward network inside transformer



inputs = layers.Input(shape=(maxlen,))

embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

x = embedding_layer(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dropout(0.1)(x)

x = layers.Dense(100, activation="relu")(x)

x = layers.Dropout(0.1)(x)

outputs = layers.Dense(5, activation="softmax")(x)



model = keras.Model(inputs=inputs, outputs=outputs)
n_epoch = 4

batch_size = 512

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(

    X_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_val, y_val)

)
viz_report(history, n_epoch)
preds = model.predict(X_test, batch_size=batch_size, verbose=1)

submission = pd.DataFrame()

submission['review_id'] = test['review_id']

submission['rating'] = [l.classes_[idx] for idx in np.argmax(preds,axis=1)]



submission.head()
create_download_link(submission)