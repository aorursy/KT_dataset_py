import numpy as np

import pandas as pd

from tqdm import tqdm

from matplotlib import pyplot as plt

from IPython.display import clear_output

from keras.models import Sequential, Model

from keras.callbacks import Callback

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Input, Flatten, GlobalMaxPool2D
root = '../input/'

HEIGHT, WIDTH = 32, 32
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []

        



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('Log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="accuracy")

        ax2.plot(self.x, self.val_acc, label="validation accuracy")

        ax2.legend()

        

        plt.show();

        

        

plot = PlotLearning()
data = pd.read_csv(root+'train.csv').values

X = np.zeros((len(data), HEIGHT, WIDTH, 3))

y = np.zeros((len(data))).astype(np.int8)
for i in tqdm(range(len(data))):

    fname, y[i] = data[i]

    X[i] = plt.imread(root+'train/train/'+fname)/255
plt.imshow(X[520])

print(y[520])
datagen = ImageDataGenerator(

    featurewise_center=True,

    samplewise_center=True,

    featurewise_std_normalization=True,

    samplewise_std_normalization=True,

    brightness_range=[0.9, 1.3],

    rotation_range=90,

    fill_mode='nearest',

    horizontal_flip=True,

    vertical_flip=True,

    preprocessing_function=None,

    validation_split=0.3,

)



datagen.fit(X)



train_gen = datagen.flow(X, y, subset='training')

val_gen = datagen.flow(X, y, subset='validation')
model = Sequential([

    Conv2D(4, (3,3), activation='relu', padding='same'),

    Conv2D(4, (3,3), activation='relu', padding='same'),

    

    BatchNormalization(),

    MaxPool2D((2,2)),

    

    Conv2D(8, (3,3), activation='relu', padding='same'),

    Conv2D(8, (3,3), activation='relu', padding='same'),

    

    BatchNormalization(),

    MaxPool2D((2,2)),

    

    Flatten(),

    

    Dense(1, activation='sigmoid')

])

model.build((None, 32, 32, 3))

model.compile(

    optimizer=Adam(lr=0.0001),

    loss='binary_crossentropy',

    metrics=['acc']

)



model.summary()
model.fit_generator(

    train_gen,

    steps_per_epoch=2000,

    epochs=100,

    callbacks=[plot],

    validation_data=val_gen,

    validation_steps=400,

    workers=8,

    max_queue_size=100,

)
model.fit(X, y, validation_split=0.3, callbacks=[plot], epochs=30)
model.save("model.h5")
data = pd.read_csv(root+'sample_submission.csv').values

X = np.zeros((len(data), 32, 32, 3))

y = np.zeros((len(data)))
for i in tqdm(range(len(data))):

    fname, _ = data[i]

    X[i] = plt.imread(root+'test/test/'+fname)/255
res = model.predict(X)
res = res.T[0]
submission = pd.DataFrame({'id':data.T[0],'has_cactus':res})
submission.head(5)
submission.to_csv('submission.csv', index=False)