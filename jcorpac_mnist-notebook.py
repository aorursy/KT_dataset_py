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
import tensorflow.keras as keras



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_features = train_data.drop(columns={'label'}).values

train_labels = keras.utils.to_categorical(train_data['label'].values)



test_features = test_data.values



train_features = keras.utils.normalize(train_features)

test_features = keras.utils.normalize(test_features)



train_features = train_features.reshape(train_features.shape[0], 28, 28, 1)

test_features = test_features.reshape(test_features.shape[0], 28, 28, 1)



print("Training Features: {}".format(train_features.shape))

print("Training Labels: {}".format(train_labels.shape))

print("Test Features: {}".format(test_features.shape))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



input_shape = train_features.shape[1:]

output_shape = train_labels.shape[1]



model = Sequential([

    Conv2D(16, (2,2), padding="same", activation="relu", input_shape=input_shape),

    MaxPooling2D((2,2)),

    Conv2D(32, (2,2), padding="same", activation="relu"),

    MaxPooling2D((2,2)),

    Conv2D(64, (2,2), padding="same", activation="relu"),

    Flatten(),

    Dense(128, input_shape=[input_shape], activation='relu'),

    Dropout(rate=0.5),

    Dense(128, activation='relu'),

    Dropout(rate=0.5),

    Dense(output_shape, activation='softmax')

])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
model.fit(train_features, train_labels, epochs=25)

#model.save('mnist_predictor.model')
predictions = model.predict([test_features])

predictions = [np.argmax(pred) for pred in predictions]



submission = pd.DataFrame()

image_id = pd.Series(list(range(1, len(predictions)+1)))

labels = pd.Series(predictions)

submission['ImageId'] = image_id

submission['Label'] = labels

submission.set_index('ImageId')



submission.to_csv('submission.csv', index=False)