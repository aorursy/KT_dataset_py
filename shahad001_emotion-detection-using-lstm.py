import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D

pd.set_option('display.max_colwidth', -1)
data = pd.read_csv("../input/twitter-text-emotions/text_emotion.csv")
print(data.head(2))
print(data['content'][:10])
#data cleaning
clean_data = []

# start cleaning data
for text in data['content']:
#     print('\nmain text=> ',text)
    tokens = word_tokenize(text) # split the tokens
    tokens = [w.lower() for w in tokens] # convert to lower case
#     print('tokens=> ',tokens)
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()] # filter all the punciations
#     print('words=> ', words)
    
    # filtering stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
#     print(stop_words)
    words = [w for w in words if not w in stop_words]
#     print('words after stop words=> ', words)
    
    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed_words = [porter.stem(word) for word in words]
#     print('stemmed words=> ', stemmed_words)
    clean_text = ''
    for w in stemmed_words:
        clean_text += ' ' + w
#     print(clean_text)
    clean_data.append(clean_text)

print('Clean data created')
data['clean_tweet'] = clean_data
# print(data['clean_tweet'])
data['tweet_len'] = data['clean_tweet'].apply(len)
# print(data['tweet_len'])  
    

# data.groupby(['tweet_len', 'sentiment']).size().unstack().plot(kind='line', stacked=False)
data['sentiment'].value_counts().plot(kind='bar', stacked=False)
# tokenization
max_words = 2000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(data['clean_tweet'].values)
X = tokenizer.texts_to_sequences(data['clean_tweet'].values)
# print(data['clean_tweet'][:5])
# print('Tokenized sentences', X[:5])
X = pad_sequences(X, maxlen=32)
# print(X[:5])
print(X.shape[1])

enbedding_out_dim = 256
lstm_out_dim = 256

model = Sequential()
model.add(Embedding(max_words, enbedding_out_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out_dim+1))
model.add(Dense(13,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
# data set to train
dummies = pd.get_dummies(data['sentiment'])
Y = dummies.values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 50)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
# print(data['sentiment'][:10], '->', Y[:10])
dict_emotion = {}
dict_label = {}
for i in range(len(Y)):
    dict_emotion[data['sentiment'][i]] = np.argmax(Y[i])
    dict_label[np.argmax(Y[i])] = data['sentiment'][i]
    if len(dict_emotion) == 13:
        print('Break at: ', i)
        break
#     print(data['sentiment'][i], '->', Y[i])
print(dict_emotion, dict_label)
X_val = X_train[:500]
Y_val = Y_train[:500]
partial_X_train = X_train[500:]
partial_Y_train = Y_train[500:]

# train the net
batch_size = 512
history = model.fit(X_train,Y_train, 
                    epochs = 50, 
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# validation
total, correct, false = 0, 0, 0
# print(len(X_val))
for x in range(len(X_val)):
    total += 1
#     print(x)

    result = model.predict(X_val[x].reshape(1, X_test.shape[1]), batch_size=1)[0]
#     print(np.argmax(result), np.argmax(Y_val[x]))

    if np.argmax(result) == np.argmax(Y_val[x]):
        correct += 1

    else:
        false += 1
print("accuracy", correct / total * 100, "%")
# print("negative accuracy", neg_correct / negative_count * 100, "%")
covid_data = pd.read_csv("../input/twitter-text-emotions/20200410_134114_covid_19_tweets.csv", usecols=[0,5], nrows=1000)
# print(covid_data.head(2))
print(covid_data['tweet_text'][:10])
#data cleaning
clean_tweet_data = []
# filtering stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import string
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

# start cleaning data
for text in covid_data['tweet_text']:
#     print('\nmain text=> ',text)
    tokens = word_tokenize(text) # split the tokens
    tokens = [w.lower() for w in tokens] # convert to lower case
#     print('tokens=> ',tokens)
    # remove punctuation from each word
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()] # filter all the punciations
#     print('words=> ', words)
    
    
#     print(stop_words)
    words = [w for w in words if not w in stop_words]
#     print('words after stop words=> ', words)
    
    # stemming of words
    
    stemmed_words = [porter.stem(word) for word in words]
#     print('stemmed words=> ', stemmed_words)
    clean_text = ''
    for w in stemmed_words:
        clean_text += ' ' + w
#     print(clean_text)
    clean_tweet_data.append(clean_text)

print('Clean data created')
covid_data['clean_tweet'] = clean_tweet_data
# print(data['clean_tweet'])
covid_data['tweet_len'] = covid_data['clean_tweet'].apply(len)
# print(covid_data['tweet_len'])  
    

# tokenization
max_words = 2000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(covid_data['clean_tweet'].values)
X = tokenizer.texts_to_sequences(covid_data['clean_tweet'].values)
# print(covid_data['clean_tweet'][:5])
# print('Tokenized sentences', X[:5])
# print(X.shape[1])
X = pad_sequences(X, maxlen=32)
# print(X[:5])

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 50)
print(X.shape[1])
# print(X_test.shape,Y_test.shape)
total, correct, false = 0, 0, 0
# print(len(X))
count = {}
for i in range(13):
    count[i] = 0
    
for i in range(len(X)):
#     print(i, '->', X[i])
    total += 1

    result = model.predict(X[i].reshape(1,X_test.shape[1]), batch_size=1)[0]
    emotion_value = np.argmax(result)
    count[emotion_value] += 1
    emotion = dict_label[emotion_value]
#     print(covid_data['tweet_text'][i], '->', emotion)

#     if np.argmax(result) == np.argmax(Y_val[x]):
#         correct += 1

#     else:
#         false += 1
for i in range(13):
    print('number of status with emotion', "'" + dict_label[i]+"'",': ', count[i])
# print("accuracy", correct / total * 100, "%")
# print("negative accuracy", neg_correct / negative_count * 100, "%")