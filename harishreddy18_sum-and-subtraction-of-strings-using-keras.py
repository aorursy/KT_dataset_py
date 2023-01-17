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
import random

import operator

import itertools

ops = { "+": operator.add, "-": operator.sub }
def generate_equations(allowed_operators, dataset_size, min_value, max_value):

    """Generates pairs of equations and solutions to them.

    

       Each equation has a form of two integers with an operator in between.

       Each solution is an integer with the result of the operaion.

    

        allowed_operators: list of strings, allowed operators.

        dataset_size: an integer, number of equations to be generated.

        min_value: an integer, min value of each operand.

        max_value: an integer, max value of each operand.



        result: a list of tuples of strings (equation, solution).

    """

    sample = []

    number_permutations = itertools.permutations(range(min_value, max_value + 1), 2)



        # Shuffle if required. The downside is we need to convert to list first

    

    number_permutations = list(number_permutations)

    random.shuffle(number_permutations)



    # If a max_count is given, use itertools to only look at that many items

    if dataset_size is not None:

        number_permutations = itertools.islice(number_permutations, dataset_size)



    # Build an equation string for each and yield to caller

    c=0

    for x, y in number_permutations:

        if c%2==0:

            a='{}+{}'.format(x, y)

        else:

            a='{}-{}'.format(x, y)

        b=eval(a)

        a=a+'$'

        b=str(b)+'$'

        sample.append((a,b))

        c+=1

        ######################################

        ######### YOUR CODE HERE #############

        ######################################

    return sample
from sklearn.model_selection import train_test_split
allowed_operators = ['+', '-']

dataset_size = 40000

data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=999)



train_set, test_set = train_test_split(data, test_size=0.2, random_state=42,shuffle=True)
word2id = {symbol:i for i, symbol in enumerate('+-1234567890')}

#word2id['<unk>']=11

word2id['<pad>']=12

word2id['$']=13

id2word = {i:symbol for symbol, i in word2id.items()}

print(word2id)
def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):

    

    X, Y = zip(*dataset)

    

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])

    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))

    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))



    return X, np.array(Y), Xoh, Yoh



def string_to_int(string, length, vocab):

    """

    Converts all strings in the vocabulary into a list of integers representing the positions of the

    input string's characters in the "vocab"

    

    Arguments:

    string -- input string, e.g. 'Wed 10 Jul 2007'

    length -- the number of time steps you'd like, determines if the output will be padded or cut

    vocab -- vocabulary, dictionary used to index every character of your "string"

    

    Returns:

    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary

    """

    

    #make lower to standardize

    string = string.lower()

    string = string.replace(',','')

    

    if len(string) > length:

        string = string[:length]

        

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    

    if len(string) < length:

        rep += [vocab['<pad>']] * (length - len(string))

    

    #print (rep)

    return rep
from keras.utils import to_categorical

Tx = 8

Ty = 5

X, Y, Xoh, Yoh = preprocess_data(train_set, word2id, word2id, Tx, Ty)
index = 3

print("Source date:", train_set[index][0])

print("Target date:", train_set[index][1])

print()

print("Source after preprocessing (indices):", X[index])

print("Target after preprocessing (indices):", Y[index])

print()

print("Source after preprocessing (one-hot):", Xoh[index])

print("Target after preprocessing (one-hot):", Yoh[index])
from keras.models import Sequential

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply,LSTMCell,RNN,BatchNormalization

from keras.layers import RepeatVector, Dense, Activation, Lambda, Reshape,TimeDistributed

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.models import load_model, Model

import keras.backend as K

from keras import metrics
def softmax(x, axis=1):

    """Softmax activation function.

    # Arguments

        x : Tensor.

        axis: Integer, axis along which the softmax normalization is applied.

    # Returns

        Tensor, output of softmax transformation.

    # Raises

        ValueError: In case `dim(x) == 1`.

    """

    ndim = K.ndim(x)

    if ndim == 2:

        return K.softmax(x)

    elif ndim > 2:

        e = K.exp(x - K.max(x, axis=axis, keepdims=True))

        s = K.sum(e, axis=axis, keepdims=True)

        return e / s

    else:

        raise ValueError('Cannot apply softmax to a tensor that is 1D')
n_a = 60

n_s = 1024
def build_model():

    """

    Builds and returns the model based on the global config.

    """

    input_shape = (Tx, len(word2id))



    model = Sequential()



    # Encoder:

    model.add(Bidirectional(LSTM(n_a), input_shape=input_shape))

    model.add(BatchNormalization())



    # The RepeatVector-layer repeats the input n times

    model.add(RepeatVector(Ty))



    # Decoder:

    model.add(Bidirectional(LSTM(n_s, return_sequences=True)))

    model.add(BatchNormalization())



    model.add(TimeDistributed(Dense(len(word2id))))

    model.add(Activation('softmax'))



    model.compile(

        loss='categorical_crossentropy',

        optimizer=Adam(lr=0.01),

        metrics=['accuracy']

    )



    return model
model=build_model()

model.summary()
### START CODE HERE ### (≈2 lines)

opt = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,decay=0.01)

#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

### END CODE HERE ###
#model.load_weights('../input/my_model_weights.h5', by_name=True)

model.fit(Xoh,Yoh, epochs=120, batch_size=128,validation_split=0.2)
model.save_weights('my_model_weights.h5')
def one_hot_to_index(vector):

    if not np.any(vector):

        return -1



    return np.argmax(vector)



def one_hot_to_char(vector):

    index = one_hot_to_index(vector)

    if index == -1:

        return ''



    return id2word[index]



def one_hot_to_string(matrix):

    return ''.join(one_hot_to_char(vector) for vector in matrix)
EXAMPLES = [i for i in train_set[0:100]]

for example,real in EXAMPLES:

    #print(example)

    source = string_to_int(example, Tx, word2id)

    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(word2id)), source))).swapaxes(0,1)

    source=np.transpose(source)

    source=source.reshape(1,Tx,len(word2id))

    

    result=model.predict(source)

    result=result.reshape(Ty,len(word2id))

    #real = string_to_int(real, Ty, word2id)

    #real = np.array(list(map(lambda x: to_categorical(x, num_classes=len(word2id)), real))).swapaxes(0,1)

    #real=real.reshape(1,Tx,len(word2id))

    #print(result)

    #print(real)

    result=one_hot_to_string(result)

    print('##########')

    print("source:", example)

    print('real:',real)

    print("output:", result)
model.evaluate(Xoh,Yoh)
Xt, Yt, Xoht, Yoht = preprocess_data(test_set, word2id, word2id, Tx, Ty)

model.evaluate(Xoht, Yoht)