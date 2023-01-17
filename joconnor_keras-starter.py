'''Credit to anokas and Oleg Panichev for various bits of code used here.'''



import numpy as np

import pandas as pd

from tqdm import tqdm

import cv2



import scipy

from sklearn.metrics import fbeta_score

from sklearn.model_selection import train_test_split



from PIL import Image



random_seed = 0

np.random.seed(random_seed)



# Load data

train_path = '../input/train-jpg/'

test_path = '../input/test-jpg/'

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/sample_submission.csv')





flatten = lambda l: [item for sublist in l for item in sublist]

labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))



label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}



def get_data(tag_df, path):

    x = []

    y = []

    for f, tags in tqdm(tag_df, miniters=1000):

        img = cv2.imread('{}/{}.jpg'.format(path, f))

        targets = np.zeros(17)

        for t in tags.split(' '):

            targets[label_map[t]] = 1

        img = img[64:191, 64:191]

        x.append(cv2.resize(img, (64, 64)))

        y.append(targets)

    return x, y
x_train, y_train = get_data(train.iloc[:10000].values, train_path)

x_test, y_test = get_data(test.values, test_path)
x_train = np.array(x_train)

y_train = np.array(y_train)

x_test = np.array(x_test)

y_test = np.array(y_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
from keras.models import Model

from keras.layers import Input

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator



def build_model():

    inp = Input((64,64,3))

    hidden = Convolution2D(32, (3,3), activation="relu")(inp)

    hidden = Convolution2D(32, (3,3), activation="relu")(hidden)

    hidden = MaxPooling2D()(hidden)

    hidden = Flatten()(hidden)

    hidden = Dense(64)(hidden)

    hidden = Dropout(0.4)(hidden)

    out = Dense(17, activation="sigmoid")(hidden)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model
batch_size=128

model = build_model()

gen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    vertical_flip=True,

    horizontal_flip=True,

    zoom_range=[1., 1.2],

)

gen.fit(x_train)

model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),

                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=4,

                    validation_data=(gen.standardize(x_val.astype(np.float32)), y_val),

                    verbose=2)
preds = model.predict(gen.standardize(x_test.astype(np.float32)))

labels = []

for row in tqdm(preds, miniters=1000):

    labels.append(" ".join(inv_label_map[i] for i, prob in enumerate(row) if prob > 0.2))

test["tags"] = labels

test.to_csv("submission.csv.gz", compression="gzip", index=False)