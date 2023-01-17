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
# importing necesarry libraries to be able to run



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.utils import np_utils
dataset = pd.read_excel("/kaggle/input/trump-comment-dataset/Trump_data.xlsx")
# The whole dataset is too big, which would make the training take over a day. So i took about 10% of the data and to reduce training time



df = dataset.drop(dataset.tail(4500).index)

df
# Merging all data into 1 string and removing all unnecessary puntuation



text = df.Comments

df.Comments = text.str.strip('.!? \n\t\r')

text = '\n'.join(df.Comments).lower()

text = text.strip('.!? \n\t\r')

len(text.split())

# Getting a list of all characters that in the dataset and enumerating the letters



characters = sorted(list(set(text)))



n_to_char = {n:char for n, char in enumerate(characters)}

char_to_n = {char:n for n, char in enumerate(characters)}



characters
X = []

Y = []

length = len(text)

seq_length = 100



for i in range(0, length-seq_length, 1):

    sequence = text[i:i + seq_length]

    label =text[i + seq_length]

    X.append([char_to_n[char] for char in sequence])

    Y.append(char_to_n[label])
X_modified = np.reshape(X, (len(X), seq_length, 1))

X_modified = X_modified / float(len(characters))

Y_modified = np_utils.to_categorical(Y)
model = Sequential()

model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(700, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(700))

model.add(Dropout(0.2))

model.add(Dense(Y_modified.shape[1], activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.fit(X_modified, Y_modified, epochs=30, batch_size=100)



# model.save_weights('Mid_model.h5')
# Loading the saved weights

model.load_weights('/kaggle/input/mid-model/Mid_model.h5')
string_mapped = X[99]

full_string = [n_to_char[value] for value in string_mapped]

# generating characters

for i in range(400):

    x = np.reshape(string_mapped,(1,len(string_mapped), 1))

    x = x / float(len(characters))



    pred_index = np.argmax(model.predict(x, verbose=0))

    seq = [n_to_char[value] for value in string_mapped]

    full_string.append(n_to_char[pred_index])



    string_mapped.append(pred_index)

    string_mapped = string_mapped[1:len(string_mapped)]
# combining text

txt=""

for char in full_string:

    txt = txt+char

txt
# Method that creates the piece of text that comes from the result



def combine_string(character_array):

    return_string = ""

    for char in character_array:

        return_string = return_string + char

    return return_string



# Getting the result of the network depending on the input string



def generate_review(starting_sequence):

    generated_review_mapped = [n_to_char[value] for value in starting_sequence]

    print(f'Starting sequence: {combine_string(generated_review_mapped)}')

    

    for i in range(4000):

        x = np.reshape(starting_sequence,(1,len(starting_sequence), 1))

        x = x / float(len(characters))



        pred_index = np.argmax(model.predict(x, verbose=0))

        seq = [n_to_char[value] for value in starting_sequence]

        generated_review_mapped.append(n_to_char[pred_index])



        starting_sequence.append(pred_index)

        starting_sequence = starting_sequence[1:len(starting_sequence)]



    return generated_review_mapped
generated_review_1 = generate_review(X[98].copy())

print(combine_string(generated_review_1))
custom_starting_sequence = list(' ussia is green and beautifull like everything around the world pakistan and big schools, president ')

custom_starting_sequence_mapped = [char_to_n[value] for value in custom_starting_sequence]



generated_custom_review = generate_review(custom_starting_sequence_mapped)

print(combine_string(generated_custom_review))