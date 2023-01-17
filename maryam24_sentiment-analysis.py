# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import tensorflow as tf

import numpy as np

import re

import seaborn as sns

from wordcloud import WordCloud



import matplotlib.pyplot as plt
class Sentiment:



    def preprocessing(string):

        emoji_pattern = re.compile("["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

        u"\U0001F1F2-\U0001F1F4"  # Macau flag

        u"\U0001F1E6-\U0001F1FF"  # flags

        u"\U0001F600-\U0001F64F"

        u"\U00002702-\U000027B0"

        u"\U000024C2-\U0001F251"

        u"\U0001f926-\U0001f937"

        u"\U0001F1F2"

        u"\U0001F1F4"

        u"\U0001F620"

        u"\u200d"

        u"\u2640-\u2642"

        "]+", flags=re.UNICODE)

        

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

        string = re.sub(r"\'s", " \'s", string)

        string = re.sub(r"\'ve", " \'ve", string)

        string = re.sub(r"n\'t", " n\'t", string)

        string = re.sub(r"\'re", " \'re", string)

        string = re.sub(r"\'d", " \'d", string)

        string = re.sub(r"\'ll", " \'ll", string)

        string = re.sub(r",", " , ", string)

        string = re.sub(r"!", " ! ", string)

        string = re.sub(r"\(", " \( ", string)

        string = re.sub(r"\)", " \) ", string)

        string = re.sub(r"\?", " \? ", string)

        string = re.sub(r"\s{2,}", " ", string)

        string = emoji_pattern.sub(r'', string)

        return string.strip().lower()

    

    

    def batch_iter(data, batch_data, num_epochs):

        

        data = np.array(data)

        data_size = len(data)

        num_batch_per_epoch = int(len(data) - 1/ batch_size )+ 1

        for epoch in range(num_epoch):

            if shuffle:

                shuffle_indices = np.random.permutation(np.arange(data_size))

                shuffled_data = data[shuffle_indices]

            else:

                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):

                start_index = batch_num * batch_size

                end_index = min((batch_num + 1) * batch_size, data_size)

                yield shuffled_data[start_index:end_index]

                

                

    def load_data_and_labels(data, max_features, maximum_len):

        

        from keras.preprocessing.text import Tokenizer

        from keras.preprocessing.sequence import pad_sequences

        from keras.utils import to_categorical

        

        train = data.sample(frac = 1).reset_index(drop=True)

        

        



        X = data['tweet'].apply(lambda x: Sentiment.preprocessing(x))

        Y = to_categorical(data['label'].values)

        tokenizer = Tokenizer(num_words=max_features)

        tokenizer.fit_on_texts(list(X))

        

        X = tokenizer.texts_to_sequences(X)

        X = pad_sequences(X, maxlen = maximum_len)

        

        

        return X, Y

        

        

        

        

        

       

    



        
train_data = pd.read_csv("/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv", encoding = "ISO-8859-1")

# positive = train_data[train_data['label']==0]

# print(train_data[train_data['target']==2])



train_data.head()
maximum_len = 300

max_features = 10000



X_train, Y_train = Sentiment.load_data_and_labels(train_data, max_features, maximum_len)

print(X_train.shape)
sns.countplot(train_data['label'])
negative = train_data[train_data['label']==0]

positive = train_data[train_data['label']==1]

                      
wordcloud = WordCloud(max_font_size = 60, max_words = 600, background_color = "black").generate(str(negative))

plt.imshow(wordcloud)
wordcloud = WordCloud(max_font_size = 60, max_words = 100, background_color = "black").generate(str(positive))

plt.imshow(wordcloud)
train_data["label"].value_counts().plot(kind = 'pie', explode = [0,0.1], figsize = (6,6), autopct = '%1.1f%%', shadow = True)



plt.ylabel("Negative and Positive")

plt.legend(["positive", "negative"])

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
from keras.layers import Input, Dense, Embedding, Flatten

from keras.layers import SpatialDropout1D

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.models import Sequential
model = Sequential()



# Input / Embdedding

model.add(Embedding(max_features, 150, input_length=maximum_len))



# CNN

model.add(SpatialDropout1D(0.2))



model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))



model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))



model.add(Flatten())



# Output layer

model.add(Dense(2, activation='sigmoid'))
epochs = 5

batch_size = 32
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=1)