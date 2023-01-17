# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
# Load Data



data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# features and labels



features = data.iloc[:, 1:].values.reshape((42000,28,28,1))

labels = data.iloc[:, 0].values.reshape((42000,1))

test = test.values.reshape((28000,28,28,1))
#train test split

X_train, X_dev, Y_train, Y_dev = train_test_split(features,

                                                   labels,

                                                   test_size = 0.025,

                                                   random_state = 0,

                                                   stratify = labels)
# feature normalization



X_train = X_train/255

X_dev = X_dev/255
#One-hot-code y values

def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)]

    return Y
# one-hot encoding



Y_train, Y_dev = convert_to_one_hot(Y_train, 10), convert_to_one_hot(Y_dev, 10)
# keras model



model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam",

             loss='categorical_crossentropy',

             metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1, batch_size=32)
score = model.evaluate(X_dev, Y_dev, batch_size=32)

score
# Submission



pred = model.predict_classes(test)
submissions = pd.DataFrame({'ImageId': np.arange(1 , 1 + test.shape[0]), 'Label': pred.astype(int)})

submissions.to_csv('./cnn1_submission.csv', index=False)