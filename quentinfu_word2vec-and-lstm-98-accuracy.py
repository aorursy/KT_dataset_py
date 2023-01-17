from IPython.display import display, set_matplotlib_formats
from collections import Counter
from itertools import chain
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import warnings
set_matplotlib_formats('svg')
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/fake-news-detection/data.csv')
df.head()
df['website'] = df.URLs.apply(lambda x: x.split('/')[2])
df.pivot_table(index = 'website', columns = 'Label', values = 'URLs', aggfunc='count').fillna(0).astype(int)
sns.set_style("white")
sns.set_palette('Set2')
fig, ax=plt.subplots(figsize=(3, 2))
ax = sns.countplot(df.Label)
fig.show()
df['text'] = df['Headline'] + " " + df['Body']
df = df.drop(columns = ['Headline', 'Body'])
df = df.loc[~df['text'].isna()] # 
stop_words = set(stopwords.words('english'))
to_remove = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
stop_words.update(to_remove)
print('Number of stopwords:', len(stop_words))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub('\[[^]]*\]', '', text)
    text = (" ").join([word for word in text.split() if not word in stop_words])
    text = "".join([char for char in text if not char in to_remove])
    return text

df['text'] = df['text'].apply(clean_text)
def print_frequency(words, name):
    counter = Counter(chain.from_iterable(words))
    df_word_distribution = pd.DataFrame(counter.values(), index = counter.keys(), columns = ['Frequency'])
    df_word_distribution.index = df_word_distribution.index.set_names(name)
    df_word_distribution = df_word_distribution.sort_values(by = 'Frequency', ascending = False, axis=0)[:15]
    df_word_distribution = df_word_distribution.pivot_table(columns = name)
    df_word_distribution = df_word_distribution.sort_values(by = 'Frequency', ascending = False, axis=1)
    display(df_word_distribution)
    
words_fake = [s.split() for s in df.loc[df.Label == 0]['text']]
words_true = [s.split() for s in df.loc[df.Label == 1]['text']]
print_frequency(words_fake, 'Fake')
print_frequency(words_true, 'True')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6,3))
text_len=df[df.Label==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len, bins = 50)
ax1.set_title('True texts')
text_len=df[df.Label==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len, bins = 50)
ax2.set_title('Fake texts')
fig.suptitle('Number of words in texts', y=0)
fig.show()
text_train, text_test, y_train, y_test = train_test_split(df['text'], df['Label'], test_size = 0.2, random_state = 42) 
size_embedding = 200 #Dimensionality of the feature vectors
windows = 2 #Maximum distance between the current and predicted word within a sentence
min_count = 1 #Ignores words with total frequency lower than this
maxlen = 1000 #Length decided for the text (adjusted by padding and truncating)

text_train_splited = [article.split() for article in text_train]
w2v_model = gensim.models.Word2Vec(sentences = text_train_splited, 
                                   size = size_embedding, 
                                   window = windows, 
                                   min_count = min_count)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train_splited)
text_train_tok = tokenizer.texts_to_sequences(text_train_splited)
word_index = tokenizer.word_index
print('Sive of vocabulary: ', len(word_index))

text_train_tok_pad = pad_sequences(text_train_tok, maxlen=maxlen)
def w2v_to_keras_weights(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, size_embedding))
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix

embedding_vectors = w2v_to_keras_weights(w2v_model, word_index)
def set_model(embedding_vectors):
    model = Sequential()
    model.add(Embedding(embedding_vectors.shape[0], 
                        output_dim=embedding_vectors.shape[1],
                        weights=[embedding_vectors], 
                        input_length=maxlen, 
                        trainable=False))
    model.add(LSTM(units=32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model
model = set_model(embedding_vectors = embedding_vectors)
model.summary()

history = model.fit(text_train_tok_pad, y_train, validation_split=0.2, epochs=30, batch_size = 32, verbose = 1)
def plot_loss_epochs(history):
    epochs = np.arange(1,len(history.history['acc']) + 1,1)
    train_acc = history.history['acc']
    train_loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    fig , ax = plt.subplots(1,2, figsize=(7,3))
    ax[0].plot(epochs , train_acc , '.-' , label = 'Train Accuracy')
    ax[0].plot(epochs , val_acc , '.-' , label = 'Validation Accuracy')
    ax[0].set_title('Train & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs , train_loss , '.-' , label = 'Train Loss')
    ax[1].plot(epochs , val_loss , '.-' , label = 'Validation Loss')
    ax[1].set_title('Train & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    fig.tight_layout()
    fig.show()
    
plot_loss_epochs(history)
model.fit(text_train_tok_pad, y_train, epochs=15, batch_size = 16, verbose = 0)
text_train_splited = [article.split() for article in text_test]
text_test_tok = tokenizer.texts_to_sequences(text_train_splited)
text_test_tok_pad = pad_sequences(text_test_tok, maxlen=maxlen)
pred = (model.predict(text_test_tok_pad) > 0.5).astype("int32")

print(classification_report(y_test, pred, target_names = ['Fake','Not Fake'])) 
cm = pd.DataFrame(confusion_matrix(y_test,pred))

fig , ax = plt.subplots(figsize = (2,2))
ax = sns.heatmap(cm, annot = True, xticklabels = ['Fake','True'] , yticklabels = ['Fake','True'], cbar = False, fmt='')
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); fig.show()