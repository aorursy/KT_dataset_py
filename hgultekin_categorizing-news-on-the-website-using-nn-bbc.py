import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

sns.set_style('darkgrid')
df = pd.read_csv('../input/bbcnewsarchive/bbc-news-data.csv', sep='\t')
df
df.info()
df.category.value_counts()
stopwords = set(stopwords.words('english'))
set(list(stopwords)[0:15]) #showing only the first 15 elements of the stopwords set
len(stopwords)
contents_clean = []
for content_clean in df.content:
    for word in stopwords:
        token = " " + word + " "
        content_clean = content_clean.replace(token, " ")
    contents_clean.append(content_clean)
df['content_clean'] = np.array(contents_clean)
df.head()
len(df.content_clean[2224])
plt.figure(figsize = (12, 6))
sns.countplot(df.category)
X = list(df.content_clean)
y = list(df.category)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42, shuffle = True)
unique_elements_train, counts_elements_train = np.unique(y_train, return_counts=True)
print("Frequency of unique labels in the train set:")
print(np.asarray((unique_elements_train, counts_elements_train)))
unique_elements_test, counts_elements_test = np.unique(y_test, return_counts=True)
print("Frequency of unique labels in the test set:")
print(np.asarray((unique_elements_test, counts_elements_test)))
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
vocab_size = 15000
embedding_dim = 32
max_length = 256
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
X_train_sqncs = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sqncs, padding=padding_type, maxlen=max_length)

X_test_sqncs = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sqncs, padding=padding_type, maxlen=max_length)

print(len(X_test_sqncs))
print(X_test_padded.shape)
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(df.category)
label_index = label_tokenizer.word_index

y_train_label_sqncs = np.array(label_tokenizer.texts_to_sequences(y_train))
y_test_label_sqncs = np.array(label_tokenizer.texts_to_sequences(y_test))

print(y_train_label_sqncs[0])
print(y_train_label_sqncs[1])
print(y_train_label_sqncs[2])
print(y_train_label_sqncs.shape)

print(y_test_label_sqncs[0])
print(y_test_label_sqncs[1])
print(y_test_label_sqncs[2])
print(y_test_label_sqncs.shape)
len(word_index)
dict(list(word_index.items())[0:15]) #showing only the first 15 elements of the word_index dictionary
label_index
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 35
history = model.fit(X_train_padded, y_train_label_sqncs, epochs=num_epochs, validation_data=(X_test_padded, y_test_label_sqncs), verbose=2)
y_pred = model.predict_classes(X_test_padded)
X_test_padded
y_pred
plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.legend(['accuracy', 'val_accuracy'])
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'])
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_content(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

reverse_label_index = dict([(value, key) for (key, value) in label_index.items()])

def decode_labels(text):
    text = np.array([text])
    return ' '.join([reverse_label_index.get(i, '?') for i in text])
X_test[1]
y_test[1]
decode_content(X_test_padded[1])
decode_labels(y_pred[1])
X_test[34]
y_test[34]
decode_content(X_test_padded[34])
decode_labels(y_pred[34])
X_test[400]
y_test[400]
decode_content(X_test_padded[400])
decode_labels(y_pred[400])
df.category.drop_duplicates().values
plt.figure(figsize=(14,10))
conf_mat = confusion_matrix(y_test_label_sqncs, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=list(label_index.keys()), yticklabels=list(label_index.keys()))
plt.ylabel('Actual')
plt.xlabel('Predicted')