# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import gensim

import os

import re

import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.callbacks import ModelCheckpoint

from keras.models import load_model
data_1 = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

data_2 = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)



data =  pd.concat([data_1, data_2])

data.head()
def clean_text(text):

    text = text.lower()

    

    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    text = pattern.sub('', text)

    text = " ".join(filter(lambda x:x[0]!='@', text.split()))

    emoji = re.compile("["

                           u"\U0001F600-\U0001FFFF"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    text = emoji.sub(r'', text)

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)        

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"where's", "where is", text) 

    text = re.sub(r"\'ll", " will", text)  

    text = re.sub(r"\'ve", " have", text)  

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"don't", "do not", text)

    text = re.sub(r"did't", "did not", text)

    text = re.sub(r"can't", "can not", text)

    text = re.sub(r"it's", "it is", text)

    text = re.sub(r"couldn't", "could not", text)

    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)

    return text
import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



def CleanTokenize(df):

    head_lines = list()

    lines = df["headline"].values.tolist()



    for line in lines:

        line = clean_text(line)

        # tokenize the text

        tokens = word_tokenize(line)

        # remove puntuations

        table = str.maketrans('', '', string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        # remove non alphabetic characters

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words("english"))

        # remove stop words

        words = [w for w in words if not w in stop_words]

        head_lines.append(words)

    return head_lines



head_lines = CleanTokenize(data)

head_lines[0:10]
from collections import Counter

from wordcloud import WordCloud, ImageColorGenerator

pos_data = data_2.loc[data_2['is_sarcastic'] == 1]

pos_head_lines = CleanTokenize(pos_data)

pos_lines = [j for sub in pos_head_lines for j in sub] 

word_could_dict=Counter(pos_lines)

# word_could_dict.most_common(10)



wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")
validation_split = 0.2

max_length = 25





tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(head_lines)

sequences = tokenizer_obj.texts_to_sequences(head_lines)



word_index = tokenizer_obj.word_index

print("unique tokens - ",len(word_index))

vocab_size = len(tokenizer_obj.word_index) + 1

print('vocab size -', vocab_size)



lines_pad = pad_sequences(sequences, maxlen=max_length, padding='post')

sentiment =  data['is_sarcastic'].values



indices = np.arange(lines_pad.shape[0])

np.random.shuffle(indices)

lines_pad = lines_pad[indices]

sentiment = sentiment[indices]



num_validation_samples = int(validation_split * lines_pad.shape[0])



X_train_pad = lines_pad[:-num_validation_samples]

y_train = sentiment[:-num_validation_samples]

X_test_pad = lines_pad[-num_validation_samples:]

y_test = sentiment[-num_validation_samples:]
print('Shape of X_train_pad:', X_train_pad.shape)

print('Shape of y_train:', y_train.shape)



print('Shape of X_test_pad:', X_test_pad.shape)

print('Shape of y_test:', y_test.shape)
embeddings_index = {}

embedding_dim = 100

GLOVE_DIR = "../input/glove-global-vectors-for-word-representation"

f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding = "utf-8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

c = 0

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        c+=1

        embedding_matrix[i] = embedding_vector

print(c)
embedding_layer = Embedding(len(word_index) + 1,

                            embedding_dim,

                            weights=[embedding_matrix],

                            input_length=max_length,

                            trainable=False)
model = Sequential()

model.add(embedding_layer)

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.25))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



print('Summary of the built model...')

print(model.summary())
history = model.fit(X_train_pad, y_train, batch_size=32, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'g', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'g', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
def predict_sarcasm(s):

    x_final = pd.DataFrame({"headline":[s]})

    test_lines = CleanTokenize(x_final)

    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)

    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    pred = model.predict(test_review_pad)

    pred*=100

    if pred[0][0]>=50: return "It's a sarcasm!" 

    else: return "It's not a sarcasm."
predict_sarcasm("You just broke my car window. Great job.")
predict_sarcasm("I was depressed. He asked me to be happy. I am not depressed anymore.")
predict_sarcasm("You just saved my dog's life. Thanks a million.")