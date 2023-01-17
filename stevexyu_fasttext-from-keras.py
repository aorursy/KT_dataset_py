# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

import numpy as np

import pandas as pd

import re



from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Embedding

from keras.layers import GlobalAveragePooling1D



from sklearn.model_selection import train_test_split
def create_ngram_set(input_list, ngram_value=2):

    """

    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)

    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)

    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]

    """

    return set(zip(*[input_list[i:] for i in range(ngram_value)]))
def add_ngram(sequences, token_indice, ngram_range=2):

    """

    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram

    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]

    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}

    >>> add_ngram(sequences, token_indice, ngram_range=2)

    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram

    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]

    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}

    >>> add_ngram(sequences, token_indice, ngram_range=3)

    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]

    """

    new_sequences = []

    for input_list in sequences:

        new_list = input_list[:]

        for ngram_value in range(2, ngram_range + 1):

            for i in range(len(new_list) - ngram_value + 1):

                ngram = tuple(new_list[i:i + ngram_value])

                if ngram in token_indice:

                    new_list.append(token_indice[ngram])

        new_sequences.append(new_list)



    return new_sequences
# Set parameters:

# ngram_range = 2 will add bi-grams features

ngram_range = 2

max_features = 20000

maxlen = 80

batch_size = 32

embedding_dims = 50

epochs = 5
train = pd.read_csv("../input/train.csv", index_col=None)

print(train.shape)



display(train.head())



test = pd.read_csv("../input/test.csv", index_col=None)

print(test.shape)



display(test.head())
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}
def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = str(x)

    x = re.sub(r'[0-9]{5,}', '#####', x)

    x = re.sub(r'[0-9]{4}', '####', x)

    x = re.sub(r'[0-9]{3}', '###', x)

    x = re.sub(r'[0-9]{2}', '##', x)

    

    return x



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    

    text = mispellings_re.sub(replace, text)

    return text
train["question_text"] = train["question_text"].apply(lambda x: x.lower())

test["question_text"] = test["question_text"].apply(lambda x: x.lower())



train["question_text"] = train["question_text"].map(clean_text)

test["question_text"] = test["question_text"].map(clean_text)



train["question_text"] = train["question_text"].map(clean_numbers)

test["question_text"] = test["question_text"].map(clean_numbers)



train["question_text"] = train["question_text"].map(replace_typical_misspell)

test["question_text"] = test["question_text"].map(replace_typical_misspell)
df = pd.concat([train ,test],sort=True)



train, dev = train_test_split(train, test_size=0.1, random_state=2019)



train_X = train["question_text"].fillna("_na_").values

dev_X = dev["question_text"].fillna("_na_").values

test_X = test["question_text"].fillna("_na_").values



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(df['question_text']))

x_train = tokenizer.texts_to_sequences(train_X)

x_test = tokenizer.texts_to_sequences(dev_X)



test_X = tokenizer.texts_to_sequences(test_X)



#train_X = pad_sequences(train_X, maxlen=maxlen)

#dev_X = pad_sequences(dev_X, maxlen=maxlen)

#test_X = pad_sequences(test_X, maxlen=maxlen)



y_train = train['target'].values

y_test = dev['target'].values
print(len(x_train), 'train sequences')

print(len(x_test), 'validation sequences')

print(len(test_X), 'test sequences')

print('Average train sequence length: {}'.format(

    np.mean(list(map(len, x_train)), dtype=int)))

print('Average validation sequence length: {}'.format(

    np.mean(list(map(len, x_test)), dtype=int)))

print('Average test sequence length: {}'.format(

    np.mean(list(map(len, test_X)), dtype=int)))
if ngram_range > 1:

    print('Adding {}-gram features'.format(ngram_range))

    # Create set of unique n-gram from the training set.

    ngram_set = set()

    for input_list in x_train:

        for i in range(2, ngram_range + 1):

            set_of_ngram = create_ngram_set(input_list, ngram_value=i)

            ngram_set.update(set_of_ngram)



    # Dictionary mapping n-gram token to a unique integer.

    # Integer values are greater than max_features in order

    # to avoid collision with existing features.

    start_index = max_features + 1

    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}

    indice_token = {token_indice[k]: k for k in token_indice}



    # max_features is the highest integer that could be found in the dataset.

    max_features = np.max(list(indice_token.keys())) + 1



    # Augmenting x_train and x_test with n-grams features

    x_train = add_ngram(x_train, token_indice, ngram_range)

    x_test = add_ngram(x_test, token_indice, ngram_range)

    test_X = add_ngram(test_X, token_indice, ngram_range)

    print('Average train sequence length: {}'.format(

        np.mean(list(map(len, x_train)), dtype=int)))

    print('Average validation sequence length: {}'.format(

        np.mean(list(map(len, x_test)), dtype=int)))

    print('Average test sequence length: {}'.format(

        np.mean(list(map(len, test_X)), dtype=int)))
print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

test_X = sequence.pad_sequences(test_X, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

print('test_X shape:', test_X.shape)
print('Build model...')

model = Sequential()



# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

model.add(Embedding(max_features,

                    embedding_dims,

                    input_length=maxlen))



# we add a GlobalAveragePooling1D, which will average the embeddings

# of all words in the document

model.add(GlobalAveragePooling1D())



# We project onto a single unit output layer, and squash it with a sigmoid:

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=2,

          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)

pred_test_y = (pred_test_y > .5).astype(int)



test['prediction'] = pred_test_y

cols = ['qid','prediction']

test=test[cols]

test.to_csv('submission.csv', index=False) 
pred_test_y
print(os.listdir("."))
!head submission.csv