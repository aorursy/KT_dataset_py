from nltk.corpus import gutenberg

import nltk

nltk.download('gutenberg')

texts = []

punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''

for id in gutenberg.fileids():

  text = gutenberg.words(id)[1:]

  text = [''.join(letter for letter in word.lower() if letter not in punctuations) for word in text if word not in punctuations and word!='']

  texts.append(text)

print(texts[0][:100])
all_words = []

for text in texts:

  all_words += text

unique_words = set(all_words)

len(unique_words)
unique_dict = dict()

for i, word in enumerate(unique_words):

  unique_dict.update({word:i})
word_lists = []

for text in texts:

  for i, word in enumerate(text):

    if i + 1 < len(text) - 1: 

      word_lists.append([word] + [text[(i + 1)]])

    if i - 1 >= 0:

      word_lists.append([word] + [text[(i - 1)]])
len(word_lists)
import numpy as np

from scipy import sparse



hot_encoded = dict()

for word in unique_words:

  encode = np.zeros(len(unique_words))

  encode[unique_dict.get(word)] = 1

  hot_encoded.update({word:sparse.csr_matrix(encode)})
X = []

Y = []





for i, word_list in enumerate(word_lists):

  if i == 10000:

    break

  print(i)



  X_row = hot_encoded[word_list[0]]

  Y_row = hot_encoded[word_list[1]]



  X = sparse.vstack([X, X_row], format='csr')

  Y = sparse.vstack([Y, Y_row], format='csr')

type(X)
X.shape
from keras.models import Input, Model

from keras.layers import Dense

import numpy as np



# Defining the neural network

inp = Input(shape=(X.shape[1],), sparse=True)

x = Dense(units=2, activation='linear')(inp)

x = Dense(units=Y.shape[1], activation='softmax')(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')



index = 256

while index < X.shape[0]:



  X_batch = X[:index].toarray()

  Y_batch = Y[:index].toarray()

  index+= 256

  model.train_on_batch(

      X_batch,

      Y_batch

      )



weights = model.get_weights()[0]



embedding_dict = {}

for word in unique_words: 

    embedding_dict.update({

        word: weights[unique_dict.get(word)]

        })
with open('output.txt', 'w') as f:

    f.write(str(len(embedding_dict)))

    f.write(' '+str(2)+'\n')

    i=0

    for k,v in embedding_dict.items():

        if i == 0:

            i+=1

            continue



        f.write(k)

        for elem in v:

            f.write(' '+str(elem))

        f.write('\n')
!pip3 install vecto
result