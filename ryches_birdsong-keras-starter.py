import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
train_df = pd.read_csv("../input/birdsong-recognition/train.csv")

test_df = pd.read_csv("../input/birdsong-recognition/test.csv")
train_df
import cv2

import matplotlib.pyplot as plt

base_path = "../input/birdsongspectrograms/"

def read_img(img_path):

    img = cv2.imread(base_path + img_path[:-3] + "jpg", 0)

    return img
train_df
batch_size = 32

img_size = (128, 1200)
num_classes = train_df["ebird_code"].nunique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df["ebird_code"] = le.fit_transform(train_df["ebird_code"])
files = [file[:-3] + "mp3" for file in os.listdir(base_path)]
train_df = train_df[train_df["filename"].isin(files)]
train_df
import keras

class DataGenerator(keras.utils.Sequence):

    def __init__(self, df=train_df, im_path = base_path, augmentations=None, batch_size=batch_size, img_size=img_size, shuffle=True):

        'Initialization'

        self.batch_size = batch_size

        self.df = df

        self.height, self.width = img_size[0], img_size[1]

        self.shuffle = shuffle

        self.augment = augmentations

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.ceil(len(self.df) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.df))]



        # Find list of IDs

        list_IDs_im = [self.df.iloc[k] for k in indexes]

        

        # Generate data

        X, y = self.data_generation(list_IDs_im)

        return X, y

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.df))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def data_generation(self, list_IDs_im):

        X = np.empty((len(list_IDs_im),self.height,self.width, 3))

        y = np.zeros((len(list_IDs_im), num_classes))

        for i, im_path in enumerate(list_IDs_im):

            im = read_img(im_path["filename"])

            if im is None:

                print("image not loaded correctly")

                im = np.zeros((self.height, self.width, 3))

            if len(im.shape)==2:

                im = np.repeat(im[...,None],3,2)

            if im.shape[1]-self.width <= 0:

                start_seq = 0

            else:

                start_seq = np.random.randint(0, im.shape[1]-self.width)

            im = im[:, start_seq:start_seq+self.width,:]

            X[i, :, :im.shape[1], :] = im

            y[i,im_path["ebird_code"]] = 1

        X = X/255.

        return X, y
train_gen = DataGenerator(df=train_df, im_path = base_path, augmentations=None, batch_size=batch_size, img_size=img_size, shuffle=True)
import matplotlib.pyplot as plt
batch = next(train_gen.__iter__())

print(np.argmax(batch[1], axis = 1))

plt.imshow(batch[0][0])

plt.show()
from keras.applications.resnet50 import ResNet50

from keras import layers

from keras.models import Model

model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_size[0], img_size[1], 3), pooling = 'max')

final_output = keras.layers.Dense(num_classes, activation = 'softmax')(model.output)

model = Model(inputs = model.input, outputs = final_output)
# def custom_model():

#     inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

#     x = layers.Conv2D(2, 3, activation="relu")(inputs)

#     x = layers.Conv2D(4, 3, activation="relu")(x)

#     x = layers.MaxPooling2D((2, 3))(x)

#     x = layers.Conv2D(8, 3, activation="relu")(x)

#     x = layers.Conv2D(16, 3, activation="relu")(x)

#     x = layers.MaxPooling2D((2, 3))(x)

#     x = layers.Conv2D(32, 3, activation="relu")(x)

#     x = layers.Conv2D(64, 3, activation="relu")(x)

#     x = layers.MaxPooling2D((2, 3))(x)

#     x = layers.Conv2D(128, 3, activation="relu")(x)

#     x = layers.Conv2D(256, 3, activation="relu")(x)

#     x = layers.MaxPooling2D((1, 2))(x)

#     x = layers.Conv2D(512, 3, activation="relu")(x)

#     x = layers.Conv2D(1024, 3, activation="relu")(x)

#     x = layers.GlobalMaxPooling2D()(x)

#     x = layers.Dense(num_classes, activation = "relu")(x)

#     outputs = layers.Dense(num_classes, activation = "softmax")(x)

#     model = keras.Model(inputs, outputs)

#     return model

# model = custom_model()
model.summary()
from keras.optimizers import Adam

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size = .2, shuffle = True)
train_gen = DataGenerator(df=train_df, im_path = base_path, augmentations=None, batch_size=batch_size, img_size=img_size, shuffle=True)

val_gen = DataGenerator(df=val_df, im_path = base_path, augmentations=None, batch_size=batch_size, img_size=img_size, shuffle=False)
from keras.callbacks import ModelCheckpoint, EarlyStopping
ckpt = ModelCheckpoint("model.h5", save_best_only=True)

es = EarlyStopping("val_categorical_accuracy", restore_best_weights = True, patience = 5)
history = model.fit_generator(generator=train_gen,

                            validation_data=val_gen,                            

                            epochs=20,verbose=1, callbacks=[ckpt, es])
pd.read_csv("../input/birdsong-recognition/example_test_audio_metadata.csv")
test_summary = pd.read_csv("../input/birdsong-recognition/example_test_audio_summary.csv")
import librosa

import cv2

#from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    # Stack X as [X,X,X]

#     X = np.stack([X, X, X], axis=-1)



    # Standardize

    mean = mean or X.mean()

    X = X - mean

    std = std or X.std()

    Xstd = X / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Normalize to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V



def build_spectrogram(path):

    y, sr = librosa.load(path)

    total_secs = y.shape[0] / sr

    M = librosa.feature.melspectrogram(y=y, sr=sr)

    M = librosa.power_to_db(M)

    M = mono_to_color(M)

    

    cv2.imwrite(path.split("/")[-1][:-4] + ".jpg", M, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    M = cv2.imread(path.split("/")[-1][:-4] + ".jpg", 0)

    M = np.repeat(M[...,None],3,2)/255.

    os.remove(path.split("/")[-1][:-4] + ".jpg")

    return M

M = build_spectrogram("../input/birdsong-recognition/example_test_audio/BLKFR-10-CPL_20190611_093000.pt540.mp3")

plt.imshow(M[:, :500])
test_path = "../input/birdsong-recognition/example_test_audio/"

test_files = os.listdir(test_path)
test_summary["seconds"] = test_summary["filename_seconds"].str.split("_").apply(pd.Series)[3]
for row in range(len(test_summary)):

    for file in test_files:

        if test_summary.iloc[row]["filename"] in file:

            fp = test_path + file

    M = build_spectrogram(fp, int(test_summary.iloc[row]["seconds"]) - 5)

    holder = np.empty((1, img_size[0], img_size[1], 3))

    holder[:, :, :M.shape[1], :] = M

    prediction = model.predict(holder)

    print(le.classes_[prediction[0] > .05], test_summary.iloc[row]["birds"])

    break
test_summary
pd.read_csv("../input/birdsong-recognition/sample_submission.csv")
model.save("first_model.h5")