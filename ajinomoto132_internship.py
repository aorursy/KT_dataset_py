import numpy as np 
import pandas as pd
import gc
from sklearn.model_selection import train_test_split

from transformers import *
import tensorflow as tf
true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
true['label'] = 0

cleansed_data = []
for data in true.text:
    if "@realDonaldTrump : - " in data:
        cleansed_data.append(data.split("@realDonaldTrump : - ")[1])
    elif "(Reuters) -" in data:
        cleansed_data.append(data.split("(Reuters) - ")[1])
    else:
        cleansed_data.append(data)

true["text"] = cleansed_data
true.head(5)
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
fake['label'] = 1

dataset = pd.concat([true, fake])
dataset = dataset.sample(frac = 1, random_state = 1).reset_index(drop = True)
dataset.head()
del true, fake
gc.collect()
# from tokenizers import (ByteLevelBPETokenizer,
#                         CharBPETokenizer,
#                         SentencePieceBPETokenizer,
#                         BertWordPieceTokenizer)
# !wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
# tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
X = np.load('../input/preparing-data-internship-0/X_train.npy')
for i in range(1,6):
    X = np.vstack([X, np.load(f'../input/preparing-data-internship-{i}/X_train.npy')])
print(X.shape)
X_train, X_val, _, __ = train_test_split(X, X, test_size = 0.2, random_state=42)
del _, __
gc.collect()
# X = np.mean(X, 1)
X.shape
# def build_model():
#     encoding_dim = 32
#     decoding_dim = 768
    
#     inp = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
#     transformer_layer = TFBertModel.from_pretrained('bert-base-uncased')
#     x = transformer_layer(inp)[0]
#     print(x.shape)
    
#     encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
#     decoded = tf.keras.layers.Dense(decoding_dim, activation='sigmoid')(encoded)
    
#     encoder = tf.keras.models.Model(inputs=[inp], outputs=[encoded])
#     autoencoder = tf.keras.models.Model(inputs=[inp], outputs=[decoded])
#     autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
#     return autoencoder, encoder
# autoencoder, encoder = build_model()
# this is the size of our encoded representations
encoding_dim = 96

# this is our input placeholder
input_ = tf.keras.layers.Input(shape=(None,256))
# "encoded" is the encoded representation of the input
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_)

# "decoded" is the lossy reconstruction of the input
decoded = tf.keras.layers.Dense(256, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = tf.keras.models.Model(input_, decoded)

# intermediate result
# this model maps an input to its encoded representation
encoder = tf.keras.models.Model(input_, encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, 
                                                          verbose = 1, min_delta = 0.0001, restore_best_weights = True)

autoencoder.fit(X_train, X_train,
                epochs=70,
                batch_size=128,
                shuffle=True,
                validation_data = (X_val, X_val),
                )
del X_train, X_val
gc.collect()
# reconst_test = autoencoder.predict(X_train)
encode_test = encoder.predict(X)
encode_test.shape
encode_test = encode_test.reshape(dataset.shape[0],200*encoding_dim)
from sklearn import cluster

# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(n_clusters=2, verbose=1)

# Fit predict will return labels
clustered = kmeans.fit_predict(X)
correct = 0
incorrect = 0
for index, row in enumerate(dataset['label'].values):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")