import pandas as pd

from keras.utils import to_categorical



train_data = pd.read_csv('../input/digit-recognizer/train.csv')

y = to_categorical(train_data.label, num_classes=10, dtype='float32')

X = train_data.drop('label', axis=1).to_numpy()



test_X = pd.read_csv('../input/digit-recognizer/test.csv').to_numpy()
from keras.models import Sequential

from keras.layers import Reshape, Dense, Dropout, Flatten, Conv2D, MaxPooling2D



model = Sequential([

    Reshape((28, 28, 1), input_shape=(784,)),

    Conv2D(32, (3, 3), activation='relu'),

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),



    Conv2D(64, (3, 3), activation='relu'),

    Conv2D(64, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),



    Flatten(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X, y, epochs=20, batch_size=128, validation_split=0.2)
test_y = model.predict(test_X)
from numpy import argmax



sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sub['Label'] = argmax(test_y, axis=1)

sub.to_csv('submission.csv', index=False)