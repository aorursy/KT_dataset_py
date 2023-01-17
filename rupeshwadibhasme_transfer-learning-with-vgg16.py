# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from PIL import Image

from skimage.transform import resize

from random import shuffle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
list_paths = []

for subdir, dirs, files in os.walk("../input"):

    for file in files:

        #print os.path.join(subdir, file)

        filepath = subdir + os.sep + file

        list_paths.append(filepath)
list_train = [filepath for filepath in list_paths if "train/" in filepath]

shuffle(list_train)

list_test = [filepath for filepath in list_paths if "test/" in filepath]



list_train = list_train

list_test = list_test

index = [os.path.basename(filepath) for filepath in list_test]
list_classes = list(set([os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_paths if "train" in filepath]))
list_classes = ['Sony-NEX-7',

 'Motorola-X',

 'HTC-1-M7',

 'Samsung-Galaxy-Note3',

 'Motorola-Droid-Maxx',

 'iPhone-4s',

 'iPhone-6',

 'LG-Nexus-5x',

 'Samsung-Galaxy-S4',

 'Motorola-Nexus-6']
def get_class_from_path(filepath):

    return os.path.dirname(filepath).split(os.sep)[-1]



def read_and_resize(filepath):

    im_array = np.array(Image.open((filepath)), dtype="uint8")

    pil_im = Image.fromarray(im_array)

    new_array = np.array(pil_im.resize((256, 256)))

    return new_array/255



def label_transform(labels):

    labels = pd.get_dummies(pd.Series(labels))

    label_index = labels.columns.values



    return labels, label_index

X_train = np.array([read_and_resize(filepath) for filepath in list_train])

X_test = np.array([read_and_resize(filepath) for filepath in list_test])



labels = [get_class_from_path(filepath) for filepath in list_train]

y, label_index = label_transform(labels)

y = np.array(y)
ROWS=256

COLS=256

from keras.models import Sequential,model_from_json

from keras.models import Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras import optimizers, losses, activations, models

from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate

from keras import applications

input_shape = (ROWS, COLS, 3)

nclass = len(label_index)



base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(ROWS, COLS,3))



add_model = Sequential()

add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(256, activation='relu'))

add_model.add(Dense(nclass, activation='softmax'))



model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)


#model = get_model()

file_path="weights.best.hdf5"



checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



early = EarlyStopping(monitor="val_acc", mode="max", patience=15)



callbacks_list = [checkpoint, early] #early



history = model.fit(X_train, y, validation_split=0.1, epochs=50, shuffle=True, verbose=2,

                              callbacks=callbacks_list)



#print(history)



model.load_weights(file_path)
predicts = model.predict(X_test)

predicts = np.argmax(predicts, axis=1)

predicts = [label_index[p] for p in predicts]



df = pd.DataFrame(columns=['fname', 'camera'])

df['fname'] = index

df['camera'] = predicts

df.to_csv("sub1.csv", index=False)