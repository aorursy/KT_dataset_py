from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import pandas as pd
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
train_df.head()
print(f'Shape of data: {train_df.shape}')
# Find the number of missing values
print(train_df.info())
# Fitting to the input data
tokenizer = Tokenizer(num_words=700, oov_token='OOV')
tokenizer.fit_on_texts(train_df.head()['text'])
# Tokens 
word_index = tokenizer.word_index
print(word_index)
# Generate text sequences
sequences = tokenizer.texts_to_sequences(train_df.head()['text'])
print(sequences)
padded = pad_sequences(sequences, padding='post')
print(padded)
# Splitting the data into 2/3 as train and 1/3 as test
X_train, X_test, y_train, y_test = train_test_split(train_df['text'], train_df['target'], test_size=0.33, random_state=42)
vocab_size = 500
embedding_dim = 16
max_length = 50
trunc_type='post'
oov_tok = "<OOV>"

# Tokenization
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train)
testing_sequences = tokenizer.texts_to_sequences(X_test)

# Padding
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)  # 
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)  # , maxlen=max_length
padded.shape
testing_padded.shape
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 10
history = model.fit(padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test))
# Getting weights from the embedding
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

# Reverse mapping function from token to word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# Combining embedding and words into a DataFrame
embedding_df = pd.DataFrame()
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    embedding_df = embedding_df.append(pd.Series({'word':word, 'em':embeddings}), ignore_index=True)
    
embedding_df = pd.concat([embedding_df['em'].apply(pd.Series), embedding_df['word']], axis=1)
# Using PCA to map 16 embedding values to 3 to plot
p = PCA(n_components=3)
principal_components = p.fit_transform(embedding_df.filter(regex=r'\d'))
embedding_df[['x', 'y', 'z']] = pd.DataFrame(principal_components)

embedding_df.shape
fig = px.scatter_3d(embedding_df, x='x', y='y', z='z', hover_name='word', color='x')
fig.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.show()