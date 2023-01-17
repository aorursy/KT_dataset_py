import pandas as pd
import numpy as np
import re
import string
from collections import Counter, namedtuple

from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.optimizers import Nadam,adam

np.random.seed(1)
data = pd.read_csv('../input/nlp-getting-started/train.csv')
data.head()
data.drop(columns = ['id','keyword','location'], inplace=True)
neg, pos = np.bincount(data.target)
print(f'Total: {len(data)} \nPositive: {pos} \nNegative: {neg}')
data.isnull().sum()
def clean_text(text):
    
    #remove urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    
    #remove html
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub(r'', text)
    
    #remove emojis
    emoji_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'',text)
    
    #remove punctuations
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)
    
    #remove stopwords
    stop = set(stopwords.words('english'))
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return ' '.join(text)
data['text'] = data['text'].apply(lambda x: clean_text(x))
data.head()
def word_counter(text):  
    
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count    

text = data['text']
counter = word_counter(text)

vocab_size = len(counter)
max_len = 20
t = Tokenizer(num_words = vocab_size)
t.fit_on_texts(data['text'])

word_index = t.word_index

dict(list(word_index.items())[:10])
df = data[:7500]
model = Sequential()
model.add(Embedding(vocab_size, 200, input_length = max_len))
model.add(LSTM(80))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

nadam = Nadam(learning_rate=0.0001)

model.compile(loss = 'binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
model.summary()
skf = StratifiedKFold(n_splits=5)
X = df['text']
y = df['target']
accuracy = []
# train model on 5 folds
for train_index, test_index in skf.split(X, y):
    
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    print("Tweet before tokenization: ", train_x.iloc[0])
    
    #Tokenize the tweets using tokenizer.
    train_tweets = t.texts_to_sequences(train_x)
    test_tweets = t.texts_to_sequences(test_x)
    print("Tweet after tokenization: ", train_tweets[0])
    
    #pad the tokenized tweet data
    train_tweets_padded = pad_sequences(train_tweets, maxlen=max_len, padding='post', truncating='post')
    test_tweets_padded = pad_sequences(test_tweets, maxlen=max_len, padding='post', truncating='post')
    print('Tweet after padding: ', train_tweets_padded[0])
    
    #train model on processed tweets
    history = model.fit(train_tweets_padded, train_y, epochs=5, validation_data = (test_tweets_padded,test_y))
    
    #make predictions
    pred_y = model.predict_classes(test_tweets_padded)
    print("Validation accuracy : ",accuracy_score(pred_y, test_y))
    
    #store validation accuracy
    accuracy.append(accuracy_score(pred_y, test_y))
print("Validation accuracy of the model :", np.mean(accuracy))
test_df = data[7501:]

tokenized_tweets = t.texts_to_sequences(test_df['text'])
padded_tweets = pad_sequences(tokenized_tweets, maxlen=max_len, padding='post', truncating='post')
test_y = test_df['target']
pred_y = model.predict_classes(padded_tweets)
accuracy_score(pred_y, test_y)
