# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from keras.preprocessing import sequence

from keras.models import Model, Input

from keras.layers import Dense, Embedding, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer

from keras.optimizers import Adam
#test=pd.read_csv('../input/test.csv')

train=pd.read_csv('../input/train.csv')



#X_train = train["comment_text"].values





#X_test = test["comment_text"].values
def one_hot_encode_by_label(dataset, new_col, col_list, threshold_val=0.1):

    

    for col in col_list:

        dataset[new_col] = (

            dataset[col] > threshold_val

        )*1.0  

    

    return dataset



#

religions = [

    'atheist',

    'buddhist',

    'christian',

    'hindu',

    'jewish',

    'muslim',

    'other_religion'

]

train = one_hot_encode_by_label(train, 'religion', religions)



#

ethnicity = [

    'asian',

    'black',

    'latino',

    'white',

    'other_race_or_ethnicity'

]

train = one_hot_encode_by_label(train, 'ethnicity', ethnicity)



# 

sexualOrientation = [

    'bisexual',

    'heterosexual',

    'homosexual_gay_or_lesbian',

    'other_gender',

    'transgender',

    'other_sexual_orientation'    

]

train = one_hot_encode_by_label(train, 'sexualOrientation', sexualOrientation)



# dataset.hist(column="religion")

# dataset.hist(column="ethnicity")

# dataset.hist(column="sexualOrientation")



train[

    train['religion'] == 1

].tail()



# # 1804874

# # len(dataset)
X = train['comment_text']

y = train[['religion','ethnicity','sexualOrientation']]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train = train[[ 'religion','ethnicity','sexualOrientation']].values
max_features = 20000  # number of words we want to keep

maxlen = 100  # max length of the comments in the model

batch_size = 64  # batch size for the model

embedding_dims = 20  # dimension of the hidden variable, i.e. the embedding dimension



tok = Tokenizer(num_words=max_features)

tok.fit_on_texts(list(X_train) + list(X_test))

x_train = tok.texts_to_sequences(X_train)

x_test = tok.texts_to_sequences(X_test)

print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')

print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))

print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

y_train=y_train[:1209265,:]
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)
comment_input = Input((maxlen,))



# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen, 

                        embeddings_initializer="uniform")(comment_input)



# we add a GlobalMaxPooling1D, which will extract features from the embeddings

# of all words in the comment

h = GlobalMaxPooling1D()(comment_emb)



# We project onto a six-unit output layer, and squash it with a sigmoid:

output = Dense(3, activation='sigmoid')(h)



model = Model(inputs=comment_input, outputs=output)



model.compile(loss='binary_crossentropy',

              optimizer=Adam(0.01),

              metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=3, validation_split=0.1)
model.evaluate(x_test,y_test)