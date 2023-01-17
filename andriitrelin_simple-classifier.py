import tensorflow as tf

import pandas as pd

import numpy as np



from tensorflow import keras

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, classification_report



import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use("ggplot")



import glob

import os
def norm_func(x, a=0, b=1):

    return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a



def normalize(x, y=None):

    x = np.apply_along_axis(norm_func, axis=1, arr=x)

    return x



data={}

for file in glob.glob("../input/cells-raman-spectra/dataset_i/**/*.csv"):

    path = file.split(os.path.sep)

    label = path[-2]

    kind = path[-1][:-4]

    if label not in data.keys():

        data[label] = {}

    data[label][kind] = normalize(pd.read_csv(file).values)
print(data.keys())

plt.plot(data['HF']['NH2'][0])

plt.plot(data['G']['COOH'][0])
train_data = {key:{} for key in data.keys()}

test_data = {key:{} for key in data.keys()}



for key in data:

    for kind in data[key]:

        train_data[key][kind], test_data[key][kind] = train_test_split(data[key][kind], test_size=0.25)
class DataGen(keras.utils.Sequence):

    def __init__(self, data, batch_size):

        self.data = data.copy()

        self.batch_size = batch_size

        self.n_classes = len(data)

        self.kinds = list(data[list(data.keys())[0]].keys())

        self.class_batch = self.batch_size//self.n_classes

        self.oh_enc = OneHotEncoder(sparse=False)

        self.oh_enc.fit(np.array(list(data.keys()))[:,np.newaxis])    



    def __len__(self):

        length=[]

        for key in self.data:

            for kind in self.data[key]:

                length.append(self.data[key][kind].shape[0])

        return min(length)//self.class_batch

    

    def __getitem__(self, idx):

        samples = dict(zip(self.kinds, [[] for i in range(len(self.kinds))]))

        labels = []

        for key in self.data:

            for kind in self.kinds:

                sample = self.data[key][kind][idx*self.class_batch:(idx+1)*self.class_batch]

                samples[kind].append(sample)

            

            labels.append(np.array([key]*self.class_batch))

        labels = np.concatenate(labels)[:,np.newaxis]

        samples = [np.concatenate(samples[key])[:,:,np.newaxis] for key in samples]

        return samples, self.oh_enc.transform(labels)



        

    def on_epoch_end(self):

        for key in self.data:

            for kind in self.data[key]:

                np.random.shuffle(self.data[key][kind])

        
class OnEpochEnd(tf.keras.callbacks.Callback):

    def __init__(self, callbacks):

        self.callbacks = callbacks



    def on_epoch_end(self, epoch, logs=None):

        for callback in self.callbacks:

            callback()
train_dg = DataGen(train_data, 64)

test_dg = DataGen(test_data, 64)

cust_callback = OnEpochEnd([train_dg.on_epoch_end, test_dg.on_epoch_end])
alpha = 0.2

drop=0.4



inp_common = Input(shape=(2090, 1))

l = Conv1D(filters=8, kernel_size=8, activation='linear')(inp_common)

l = BatchNormalization()(l)

l = LeakyReLU(alpha)(l)

l = MaxPool1D(pool_size=2)(l)



l = Conv1D(filters=16, kernel_size=16, activation='linear')(l)

l = BatchNormalization()(l)

l = LeakyReLU(alpha)(l)

out_common = MaxPool1D(pool_size=2, name="nh2_output")(l)



common_conv = Model(inp_common, out_common)



inp1 = Input(shape=(2090, 1), name="nh2_input")

inp2 = Input(shape=(2090, 1), name="cooh_input")

inp3 = Input(shape=(2090, 1), name="cooh2_input")





out1 = common_conv(inp1)

out2 = common_conv(inp2)

out3 = common_conv(inp3)



x = concatenate([out1, out2, out3])

x = Flatten()(x)



x = Dense(64, activation='linear')(x)

x = LeakyReLU(alpha)(x)

x = Dropout(drop)(x)



x = Dense(32, activation='linear')(x)

x = LeakyReLU(alpha)(x)

x = Dropout(drop)(x)

out = Dense(len(data), activation='softmax', name="main_output")(x)



model = Model(inputs=[inp1, inp2, inp3], outputs=[out])

model.compile(optimizer="adam", loss="categorical_crossentropy",

              metrics=['accuracy'])
model.fit(train_dg, epochs=75, validation_data=test_dg, callbacks=[cust_callback])
X = []

Y = []

for i in range(100):

    for x, y in test_dg:

        X.append(x)

        Y.append(y)

    test_dg.on_epoch_end()

X = [arr for arr in np.concatenate(X, axis=1)]

Y = np.concatenate(Y)
pred = model.predict(X)

print(classification_report(Y, (pred > 0.5).astype(int), target_names = train_dg.oh_enc.categories_[0], digits=3))