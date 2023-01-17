import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Flatten

from keras.models import Sequential
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

sample_df = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

n_classes = train_df['label'].max()

sample_df.head(10)
def one_hot_encode(array):

    output_array = []

    n_labels = max(array)

    for value in array:

        output_array.append(np.zeros(n_labels))

        output_array[-1][value-1] = 1

    return np.array(output_array)



def one_hot_decode(array):

    return [np.argmax(one_hot_val) + 1 for one_hot_val in array]
y_train = np.array(train_df['label'])

y_train = one_hot_encode(y_train)

train_df = train_df.drop(columns=['label'])

ids = test_df['id']

test_df = test_df.drop(columns=['id'])

print(train_df.shape)

print(test_df.shape)



scaler = MinMaxScaler()

train_df = scaler.fit_transform(train_df)

test_df = scaler.transform(test_df)



test_input = np.array(test_df).reshape((-1, 28, 28, 1))

train_input = np.array(train_df).reshape((-1, 28, 28, 1))
model = Sequential()

model.add(Conv2D(64, (4,4), input_shape=train_input.shape[1:]))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Conv2D(128, (4,4), input_shape=train_input.shape[1:]))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Flatten())

model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=train_input, y=y_train, epochs=50, batch_size=60, verbose=2)
predictions = model.predict(test_input)
submission = pd.DataFrame()

submission['id'] = pd.Series(ids, name='id')

submission['label'] = pd.Series(one_hot_decode(predictions), name='label')

display(submission)

submission.to_csv('submission.csv', index=False)