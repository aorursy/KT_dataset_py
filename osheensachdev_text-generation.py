book = []

with open('../input/internet_archive_scifi_v3.txt') as pdf:

    for line in pdf:

        book.append(line)

book[0] = book[0][:len(book[0])//1000]


import string

punctuations = string.punctuation

punctuations += '1234567890'

eol = '.!?'



cleaned_book = []

for line in book:

    cleaned_line = ''

    for char in line:

        if char in eol:

            cleaned_line += ' . '

            continue

        if char in punctuations or char == '\n':

            continue

        cleaned_line += char

    cleaned_line = cleaned_line.lower()

    cleaned_book.append(cleaned_line)



all_text = ' \n '.join(cleaned_book)

print(all_text[:200])
import numpy as np

text_tokens = all_text.split()

text_tokens = np.array(text_tokens)

text_tokens = text_tokens.reshape(len(text_tokens), 1)

print(text_tokens.shape)

print(text_tokens[:50])
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse = False, categories = 'auto', handle_unknown = 'ignore')

one_hot_encodings = encoder.fit_transform(text_tokens)
print(one_hot_encodings.shape)

print(encoder.categories_)
X_train = []

Y_train = []



group = 1



for i in range(one_hot_encodings.shape[0] - group - 1):

    X_train.append(one_hot_encodings[i:i + group].reshape(one_hot_encodings.shape[1] * group,))

    Y_train.append(one_hot_encodings[i + group + 1])

X_train = np.array(X_train)

Y_train = np.array(Y_train)

# X_train = one_hot_encodings[:-1]

# Y_train = one_hot_encodings[1:]
from keras.layers import Input, Dense, Flatten, Conv1D, Embedding, Dropout, GlobalMaxPooling1D, Activation

from keras.models import Sequential



def get_simple_model(input_size, output_size):

    model = Sequential()

    model.add(Dense(30, activation = 'relu', input_shape = [input_size]))

    model.add(Dense(10, activation = 'relu'))

    model.add(Dense(10, activation = 'relu'))

    model.add(Dense(30, activation = 'relu'))

    model.add(Dense(output_size, activation = 'softmax'))

    return model



model = get_simple_model(X_train[0].shape[0], Y_train[0].shape[0])

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 1)
prev = [['the']]

print(prev[0][0], end = ' ')

np.random.seed(0)

for i in range(300):

    prev = np.array(prev)

    encoded = encoder.transform(prev)

    encoded = encoded.reshape(1, encoded.shape[1] * group)

    prediction = model.predict(encoded)

    choice = np.random.choice(np.arange(0, prediction.shape[1]), p = prediction[0])

    word = [[encoder.categories_[0][choice]]]

    print(word[0][0], end = ' ')

    word = np.array(word)

    prev = [[word[0][0]]]




