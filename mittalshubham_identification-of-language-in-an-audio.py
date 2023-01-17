from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, RepeatVector, Reshape, Concatenate, UpSampling2D

import numpy as np

import librosa

import os

from sklearn.model_selection import train_test_split
classes = ["de", "en", "es"]

label = np.array([1, 0, 0])

files = []

y = []

for cls in classes:

    path = '../input/spoken-languages/train/' + cls + '/'

    names = os.listdir(path)

    for pos in range(len(names)):

        files.append(path + names[pos])

        y.append(label)

    label = np.roll(label, 1)



files = np.array(files)

y = np.array(y)

paths_train, paths_valid, y_train, y_valid = train_test_split(files, y, test_size = 0.2, random_state=45)

print(paths_train.shape, y_train.shape, paths_valid.shape, y_valid.shape)
def batch_generator(for_train, batch_size):

    while True:

        if for_train is True:

            idx = np.random.randint(0, paths_train.shape[0], batch_size)

        else:

            idx = np.random.randint(0, paths_valid.shape[0], batch_size)

            

        x = np.zeros((batch_size, 39, 1001, 1))

        y = np.zeros((batch_size, 3))

        for i in range(batch_size):

            if for_train is True:

                audio, sr = librosa.load(paths_train[idx[i]], sr=16000)

                y[i] = y_train[idx[i]]

            else:

                audio, sr = librosa.load(paths_valid[idx[i]], sr=16000)

                y[i] = y_valid[idx[i]]

                

            mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

            if mfcc.shape[1] < 1001:

                mfcc = np.concatenate((mfcc, np.zeros((mfcc.shape[0], 1001-mfcc.shape[1]))), axis=1)

            else:

                mfcc = mfcc[:, 0:1001]

            x[i, :, :, 0] = mfcc[1:]

            

        yield x, y
input = Input(shape=(39, 1001, 1))
temp = Conv2D(24, (6, 6), activation='relu')(input)

temp = AveragePooling2D((2,2))(temp)

temp = Conv2D(24, (6, 6), activation='relu')(temp)

temp = AveragePooling2D((2,2))(temp)

temp = Conv2D(24, (6, 6), activation='relu')(temp)

temp = AveragePooling2D((1,141))(temp)

output = Dense((3), activation='softmax')(temp)

output = Reshape((3,))(output)

print(output)
model = Model(inputs=input, outputs=output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
train_gen = batch_generator(True, batch_size=128)

valid_gen = batch_generator(False, batch_size=128)
model.load_weights("../input/identification-of-language-in-an-audio/weights.h5")
model.fit_generator(generator=train_gen,

    epochs=5,

    steps_per_epoch=paths_train.shape[0] // 128,

    validation_data=valid_gen,

    validation_steps=paths_valid.shape[0] // 128)

model.save_weights("weights.h5")
classes = ["de", "en", "es"]

label = np.array([1, 0, 0])



total = 0



for cls in classes:

    path = '../input/spoken-languages/test/' + cls + "/"

    total = total + len(os.listdir(path))



x = np.zeros((total, 39, 1001, 1))

y = np.zeros((total, 3))

num = 0

for cls in classes:

    path = '../input/spoken-languages/test/' + cls + "/"

    names = os.listdir(path)

    n = len(names)

    for pos in range(n):

        name = names[pos]

        audio, sr = librosa.load(path + '/' + name, sr=16000)

        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

        if mfcc.shape[1] < 1001:

            mfcc = np.concatenate((mfcc, np.zeros((mfcc.shape[0], 1001-mfcc.shape[1]))), axis=1)

        else:

            mfcc = mfcc[:, 0:1001]

        x[num, :, :, 0] = mfcc[1:]

        y[num] = label

        num = num + 1

    label = np.roll(label, 1)

    

print(model.evaluate(x, y)[1]*100)