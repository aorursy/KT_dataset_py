import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras import Sequential

from keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import PorterStemmer



import re

from string import punctuation



from wordcloud import WordCloud
df = pd.read_csv("../input/fake-news/train.csv", index_col = 'id')
df.head()
df.isnull().sum()
df = df.dropna()
df.shape
df['whole_text'] = df['title'] + " " + df['text']
df.head()
stop_words = stopwords.words('english')

ps = PorterStemmer()
def preprocess(text):



    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower().split()

    text = [ps.stem(word) for word in text if word not in stop_words]

    text = ' '.join(text)

    text = ''.join(p for p in text if p not in punctuation)

    

    return text
df['clean'] = df['whole_text'].apply(preprocess)
df['clean'][0]
plt.figure(figsize = (8, 8))

sns.countplot(y = y)
plt.figure(figsize=(20,20))

wordCloud = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df["label"] == 0]["clean"]))

plt.imshow(wordCloud, interpolation = 'bilinear')
plt.figure(figsize=(20,20))

wordCloud = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df["label"] == 1]["clean"]))

plt.imshow(wordCloud, interpolation = 'bilinear')
y = df['label'] # Target Column



df = df.drop(['label', 'author'], axis = 1)
X_train = df['clean']



y_train = np.asarray(y)
vocab_size = 20000

embedding_dim = 120



tokenizer = Tokenizer(num_words = vocab_size)

tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim))

model.add(Bidirectional(LSTM(128)))

model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 5)
df_test = pd.read_csv("../input/fake-news/test.csv")



test_id = df_test['id']
df_test = df_test.drop(['id', 'author'], axis = 1)

df_test.head()
df_test.shape
df_test['whole_text'] = df_test['title'] + " " + df_test['text']
df_test.fillna(method = 'ffill', inplace = True)
df_test.isnull().sum()
df_test['clean'] = df_test['whole_text'].apply(preprocess)
X_test = df_test['clean']
test_sequences = tokenizer.texts_to_sequences(X_test)

padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post') 
pred = model.predict_classes(padded_test)



pred
sub=[]

for i in pred:

    sub.append(i[0])
submission = pd.DataFrame({'id':test_id, 'label':sub})

submission.shape
submission.head()
submission.to_csv('submission.csv',index=False)