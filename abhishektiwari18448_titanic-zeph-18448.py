import pandas as pd

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout



train = pd.read_csv('../input/train.csv')



Xtrain = train.iloc[:,[2,5,6,7,9]].values

ytrain = train.iloc[:,1].values



simpleclassifier = Sequential()

simpleclassifier.add(Dense(64, activation='relu', input_dim=5))

simpleclassifier.add(Dropout(0.5))

simpleclassifier.add(Dense(32, activation='relu'))

simpleclassifier.add(Dropout(0.2))

simpleclassifier.add(Dense(16, activation='relu'))

simpleclassifier.add(Dense(1, activation='sigmoid'))

simpleclassifier.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

simpleclassifier.fit(Xtrain, ytrain, batch_size=128, epochs=30)
