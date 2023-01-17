import numpy as np
import pandas as pd
import keras
import seaborn as sns
import os
from scipy import signal
print(os.listdir("../input"))
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")
train_df.head()
sns.countplot(data = train_df, x = "is_turkey")
train_df["audio_embedding"] = train_df["audio_embedding"].apply(lambda x: np.asarray(x).reshape(-1))
plt.plot(train_df["audio_embedding"].iloc[0])
train_df["audio_embedding"] = train_df["audio_embedding"].apply(lambda x: x - x.mean())
plt.plot(train_df["audio_embedding"].iloc[0])
def spect(i):
    f, t, Sxx = signal.spectrogram(np.array(train_df["audio_embedding"].iloc[i]).reshape(-1))
    my_dpi = 100
    plt.figure(figsize=(525/my_dpi, 783/my_dpi), dpi=my_dpi)
    plt.pcolormesh(t, f, Sxx)
    plt.axis('off')
    plt.savefig("./{}.png".format(i), bbox_inches='tight', dpi=my_dpi, frameon='false')
    plt.clf()
    plt.close('all')
    img_file = scipy.misc.imresize(arr=plt.imread("./{}.png".format(i)), size=(640, 465, 3))
    img_arr = np.asarray(img_file)
    return img_arr
with Pool(2) as p:
    f = list(tqdm(p.imap(spect, range(train_df.shape[0])), total=train_df.shape[0]))
import subprocess as sp
plt.imshow(plt.imread("./0.png"))
plt.axis("off")
plt.show()
plt.imshow(plt.imread("./1.png"))
plt.axis("off")
plt.show()
plt.imshow(plt.imread("./2.png"))
plt.axis("off")
plt.show()
sp.getoutput("rm -rf *.png")

plt.imshow(f[0])
plt.axis("off")
plt.show()
from keras.layers import Dense, MaxPool2D, Conv2D, Reshape, Input, BatchNormalization, Flatten
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint

def mymodel():
    inp = Input(shape=(640, 465, 4,))
    k = BatchNormalization()(inp)
    k = Conv2D(32, (7,7), padding="same",activation="relu",strides=(2,2))(k)
    k = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(k) 
    k = Conv2D(32, (3,3), padding="same",activation="relu",strides=(1,1))(k)
    k = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(k)
    k = Conv2D(32, (3,3), padding="same",activation="relu")(k)
    k = Conv2D(32, (3,3), padding="same",activation="relu")(k)
    k = MaxPool2D(pool_size=(2, 2), padding="same",strides=(1,1))(k)
    k = Flatten()(k)
    y = Dense(2,activation="softmax")(k)
    model = Model(inp, y)
    opt = optimizers.Adam(lr=0.01,decay=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model
model = mymodel()
model.summary()
filepath = "./weight_tr5.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(np.asarray(f),
         pd.get_dummies(train_df['is_turkey']),
         epochs = 100,
         batch_size = 128,
         validation_split=0.2,
         callbacks = callbacks_list,
         verbose = 1)



















