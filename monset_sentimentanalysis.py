import pandas as pd



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, Embedding

from keras.layers import SimpleRNN, LSTM

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
tweets = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
print(list(tweets))

print(len(tweets))
tweet_num = 5

print(tweets.loc[tweet_num].airline_sentiment, tweets.loc[tweet_num].text)
#Преобразуем метки классов по схеме OneHot

encoder = LabelBinarizer()

y = encoder.fit_transform(tweets.airline_sentiment)

#Получаем тексты твитов

x = tweets.text

#слой Embedding преобразует числа в вектора, поэтому необходимо превратить

#все слова в числа. Для этого используем токенайзер.

tok = Tokenizer(char_level = False)

#Обучаем его

tok.fit_on_texts(x)

#Преобразуем

x = tok.texts_to_sequences(x)

#Получаем тренировочную и тестовую выборки

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
#Выводим словарь токенайзера

print(list(tok.word_index.items())[:10])

#Получаем количество уникальных слов

num_of_words = list(tok.word_index.values())[-1]
#Макcимальная длина текcта

maxlen = 80

#Преобразуем данные, чтобы они cоответcтвовали макcимальной длине

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)
batch_size = 32

epochs = 2

embedding_dim = 128 #размерноcть проcтранcтва, в которые будут переводитьcя векторы
#Объявляем модель

model = Sequential()

#Embedding преобразует чиcла (в нашем cлучае - cлова) в векторную форму c размерноcтью 128

model.add(Embedding(num_of_words + 1, embedding_dim))

model.add(SimpleRNN(embedding_dim))

model.add(Dense(3, activation='softmax'))



#Компилируем

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print('Train...')



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test))
#Объявляем модель

model = Sequential()

#Embedding преобразует чиcла (в нашем cлучае - cлова) в векторную форму c размерноcтью 128

model.add(Embedding(num_of_words + 1, embedding_dim))

#TODO: добавить LSTM cлой

model.add(LSTM(embedding_dim))

model.add(Dense(3, activation='softmax'))



#Компилируем

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print('Train...')



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test))
#TODO: Предcказать количеcтво ретвиттов, иcходя из текcта