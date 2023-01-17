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
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from skimage.transform import rescale
from sklearn import preprocessing
from keras.models import load_model
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
img_rows = 40
img_cols = 40
#X = np.load("../input/rescalinginputdata/40x40_rescaled_trainX.npy")
encoder = preprocessing.LabelEncoder()
encoder.fit(np.load("../input/tmpfory/trainY.npy"))
T = np.load("../input/inputtestdata40x40/testT.npy")
model = Sequential()

model.add(Conv2D(64, kernel_size = 3, activation='relu', input_shape = (img_rows, img_cols, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1000, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=8,
        zoom_range = 0.10,
        width_shift_range=0.1, 
        height_shift_range=0.1)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
#X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X, y, test_size = 0.1)
X_train2 = np.load("../input/rescalinginputdata/X_train2.npy")
X_val2 = np.load("../input/rescalinginputdata/X_val2.npy")
Y_train2 = np_utils.to_categorical(encoder.transform(np.load("../input/rescalinginputdata/Y_train2.npy")))
Y_val2 = np_utils.to_categorical(encoder.transform(np.load("../input/rescalinginputdata/Y_val2.npy")))
history = model.fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
    epochs = 18, steps_per_epoch = X_train2.shape[0]//64,  
    validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0, initial_epoch=0)
str(history.history)
history = history.history
file = open('info.csv', 'w')
file.write(str(history))
file.close()
# epochs = 20
# history = [0] * epochs

# for j in range(epochs):
#     #X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X, y, test_size = 0.1)
#     history[j] = model.fit_generator(datagen.flow(X, y, batch_size=64),
#         epochs = j+1, steps_per_epoch = X.shape[0]//64,
#         callbacks=[annealer], verbose=0, initial_epoch=j)
#     if j % 3 == 0:
#         predictions = model.predict(T)
#         predictions = np.argmax(predictions, axis=1)
#         res = encoder.inverse_transform(predictions)
#         file = open('ans' + str(j) + '.csv', 'w')
#         file.write(str(history[j].history) + '\n')
#         file.write('Id,Category\n')
#         for i in range(res.size):
#             file.write('' + str(i + 1) + ',' + str(res[i]) + '\n')
#         file.close()
#     #print("Epoch={}, Train accuracy={2:.5f}".format(j+1, max(history[j].history['acc'])))
# last = model.fit(X, y, batch_size=64, epochs=1, verbose=0, initial_epoch=0, validation_split=0.1)
# history.append(last)
#history[0].history
predictions = model.predict(T)
predictions = np.argmax(predictions, axis=1)
res = encoder.inverse_transform(predictions)
file = open('ansEpochs.csv', 'w')
file.write('Id,Category\n')
for i in range(res.size):
    file.write('' + str(i + 1) + ',' + str(res[i]) + '\n')
file.close()
# from IPython.display import HTML
# import pandas as pd
# import numpy as np
# import base64

# # function that takes in a dataframe and creates a text link to  
# # download it (will only work for files < 2MB or so)

# def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
#     df = pd.DataFrame(df)
#     df.index += 1
#     csv = df.to_csv(index=True)
#     b64 = base64.b64encode(csv.encode())
#     payload = b64.decode()
#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
#     html = html.format(payload=payload,title=title,filename=filename)
#     return HTML(html)

# create_download_link(res, filename="answer.csv")
