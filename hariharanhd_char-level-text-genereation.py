import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import RegexpTokenizer
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# Loading the dataset
data = pd.read_json('/kaggle/input/quotes-dataset/quotes.json')
print(data.shape)
data.head()
# Dropping duplicates and creating a list containing all the quotes
quotes = data['Quote'].drop_duplicates()
print(f"Total Unique Quotes: {quotes.shape}")

# Considering only top 5000 quotes
quotes_filt = quotes[:4000]
print(f"Filtered Quotes: {quotes_filt.shape}")
all_quotes = list(quotes_filt)
all_quotes[:2]
# Converting the list of quotes into a single string
processed_quotes = " ".join(all_quotes)
processed_quotes = processed_quotes.lower()
processed_quotes[:100]
# Tokeinization
tokenizer = Tokenizer(char_level=True, oov_token='<UNK>')
seq_len = 100  #(Length of the training sequence)
sentences = []
next_char = []

# Function to create the sequences
def generate_sequences(corpus):
    tokenizer.fit_on_texts(corpus)
    total_vocab = len(tokenizer.word_index) + 1
    print(f"Total Length of characters in the Text: {len(corpus)}")
    print(f"Total unique characters in the text corpus: {total_vocab}")
    
    # Loop through the entire corpus to create input sentences of fixed length 100 and the next character which comes next with a step of 1
    for i in range(0, len(corpus) - seq_len, 1):
        sentences.append(corpus[i:i+seq_len])
        next_char.append(corpus[i+seq_len])
        
            
    return sentences, next_char, total_vocab

# Generating sequences
sentences, next_char, total_vocab = generate_sequences(processed_quotes)

print(len(sentences))
print(len(next_char))
print(sentences[:1])
print(next_char[:1])
# Create a matrix of required shape
X_t = np.zeros((len(sentences), seq_len, total_vocab), dtype=np.bool)
y_t = np.zeros((len(sentences), total_vocab), dtype=np.bool)

# Loop through each sentences and each character in the sentence and replace the respective position with value 1.
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X_t[i, t, tokenizer.word_index[char]] = 1
        y_t[i, tokenizer.word_index[next_char[i]]] = 1
print(f"The shape of X: {X_t.shape}")
print(f"The shape of Y: {y_t.shape}")
print(f"Corpus Length: {len(processed_quotes)}")
print(f"Vocab Length: {total_vocab}")
print(f"Total Sequences: {len(sentences)}")
# Building the model
def create_model():
    model = Sequential()
    model.add(layers.LSTM(256, dropout=0.5, input_shape = (X_t.shape[1], X_t.shape[2])))
    model.add(layers.Dense(total_vocab, activation='softmax'))
    
    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model()
model.summary()
import gc
gc.collect()
# Training the model
callback = EarlyStopping(monitor='loss', patience=2)
model.fit(X_t, y_t, epochs=100, batch_size=64, callbacks=[callback])
model.save("char_text_gen_quotesV2.h5")
# # Loading the model
# from keras.models import load_model

# char_model = load_model("../input/char-trained-model/char_text_gen_quotesV2.h5")
# char_model.summary()
# Prediction sampling
def sample(preds, temperature = 0.5):
    preds = np.asarray(preds).astype("float64")
    scaled_pred = np.log(preds)/temperature
    scaled_pred = np.exp(scaled_pred)
    scaled_pred = scaled_pred/np.sum(scaled_pred)
    scaled_pred = np.random.multinomial(1, scaled_pred, 1)
    return np.argmax(scaled_pred)
start_index = np.random.randint(0, len(processed_quotes) - seq_len - 1)
generated = ""
sentence = processed_quotes[start_index : start_index + seq_len].lower()

for i in range(1000):
    x_pred = np.zeros((1, seq_len, total_vocab))
    for t, char in enumerate(sentence):
        x_pred[0, t, tokenizer.word_index[char]] = 1.0
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = tokenizer.index_word[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char
    
print(generated)
