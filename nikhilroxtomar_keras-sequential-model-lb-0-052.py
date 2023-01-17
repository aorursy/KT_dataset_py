import numpy as np

import pandas as pd

from keras.preprocessing import text, sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from sklearn.model_selection import train_test_split
list_of_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

max_features = 20000

max_text_length = 400

embedding_dims = 50

filters = 250

kernel_size = 3

hidden_dims = 250

batch_size = 128

epochs = 1
train_df = pd.read_csv('../input/train.csv')

print(train_df.head())
print(train_df.iloc[0, -7])

print(train_df.iloc[0, 1])
print(np.where(pd.isnull(train_df)))
x = train_df['comment_text'].values

print(x)
print("properties of x")

print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of each element {} bytes".format(type(x), x.ndim, x.shape, x.size, x.dtype, x.itemsize))
y = train_df[list_of_classes].values

print(y)
print("properties of y")

print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of each element {} bytes".format(type(y), y.ndim, y.shape, y.size, y.dtype, y.itemsize))
x_tokenizer = text.Tokenizer(num_words=max_features)

print(x_tokenizer)

x_tokenizer.fit_on_texts(list(x))

print(x_tokenizer)

x_tokenized = x_tokenizer.texts_to_sequences(x) #list of lists(containing numbers), so basically a list of sequences, not a numpy array

#pad_sequences:transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array of shape 

x_train_val = sequence.pad_sequences(x_tokenized, maxlen=max_text_length)
print("properties of x_train_val")

print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of each element {} bytes".format(type(x_train_val), x_train_val.ndim, x_train_val.shape, x_train_val.size, x_train_val.dtype, x_train_val.itemsize))
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.1, random_state=1)
print('Build model...')

model = Sequential()



# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

model.add(Embedding(max_features,

                    embedding_dims,

                    input_length=max_text_length))

model.add(Dropout(0.2))



# we add a Convolution1D, which will learn filters

# word group filters of size filter_length:

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

# we use max pooling:

model.add(GlobalMaxPooling1D())



# We add a vanilla hidden layer:

model.add(Dense(hidden_dims))

model.add(Dropout(0.2))

model.add(Activation('relu'))



# We project onto 6 output layers, and squash it with a sigmoid:

model.add(Dense(6))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()
model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

validation_data=(x_val, y_val))
test_df = pd.read_csv('../input/test.csv')

print(test_df.head())
print(np.where(pd.isnull(test_df)))
test_df.iloc[52300, 1]
x_test = test_df['comment_text'].fillna('comment_missing').values

print(x_test)
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)

x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)
y_testing = model.predict(x_testing, verbose = 1)
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[list_of_classes] = y_testing

sample_submission.to_csv("toxic_comment_classification.csv", index=False)