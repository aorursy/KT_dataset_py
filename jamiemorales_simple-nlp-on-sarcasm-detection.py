# Set-up libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
# Check source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load data
df = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)
df.head()
# Look at some details
df.info()
# Look at breakdown of label
df['is_sarcastic'].value_counts()
# Split data into 80% training and 20% validation
sentences = df['headline']
labels = df['is_sarcastic']

train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=0)

print(train_sentences.shape)
print(val_sentences.shape)
print(train_labels.shape)
print(val_labels.shape)
# Tokenize and pad
vocab_size = 10000
oov_token = '<00V>'
max_length = 100
padding_type = 'post'
trunc_type = 'post'


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# Build and train neural network
embedding_dim = 16
num_epochs = 10
batch_size = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

history = model.fit(train_padded, train_labels, batch_size=batch_size, epochs=num_epochs, 
                    verbose=2)
# Apply neural network
val_loss, val_accuracy = model.evaluate(val_padded, val_labels)
print('Val loss: {}, Val accuracy: {}'.format(val_loss, val_accuracy*100))

quick_test_sentence = [
    'canada is flattening the coronavirus curve',
    'canucks take home the cup',
    'safety meeting ends in accident'
    
]

quick_test_sequences = tokenizer.texts_to_sequences(quick_test_sentence)
quick_test_padded = pad_sequences(quick_test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
quick_test_sentiments = model.predict(quick_test_padded)

for i in range(len(quick_test_sentiments)):
    print('\n' + quick_test_sentence[i])
    if 0 < quick_test_sentiments[i] < .50:
        print('Unlikely sarcasm. Sarcasm score: {}'.format(quick_test_sentiments[i]*100))
    elif .50 < quick_test_sentiments[i] < .75:
        print('Possible sarcasm. Sarcasm score: {}'.format(quick_test_sentiments[i]*100))
    elif .75 >  quick_test_sentiments[i] < 1:
        print('Sarcasm. Sarcasm score:  {}'.format(quick_test_sentiments[i]*100))
    else:
        print('Not in range')