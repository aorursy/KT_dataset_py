from matplotlib import pyplot as plt

import math

import numpy as np



lrs = np.zeros((1000, ))

for epoch in range(1000):

    cos_inner = (math.pi * (epoch % 100)) / 100

    lrs[epoch] = 0.01 / 2 * (math.cos(cos_inner) + 1)

    

plt.figure(1, figsize=(16, 8))

plt.xlabel('epoch')

plt.ylabel('learning rate')

plt.plot(lrs)

plt.show()
from keras.callbacks import Callback

from keras import backend

from keras.models import load_model



# this callback applies cosine annealing, saves snapshots and allows to load them

class SnapshotEnsemble(Callback):

    

    __snapshot_name_fmt = "snapshot_%d.hdf5"

    

    def __init__(self, n_models, n_epochs_per_model, lr_max, verbose=1):

        """

        n_models -- quantity of models (snapshots)

        n_epochs_per_model -- quantity of epoch for every model (snapshot)

        lr_max -- maximum learning rate (snapshot starter)

        """

        self.n_epochs_per_model = n_epochs_per_model

        self.n_models = n_models

        self.n_epochs_total = self.n_models * self.n_epochs_per_model

        self.lr_max = lr_max

        self.verbose = verbose

        self.lrs = []

 

    # calculate learning rate for epoch

    def cosine_annealing(self, epoch):

        cos_inner = (math.pi * (epoch % self.n_epochs_per_model)) / self.n_epochs_per_model

        return self.lr_max / 2 * (math.cos(cos_inner) + 1)



    # when epoch begins update learning rate

    def on_epoch_begin(self, epoch, logs={}):

        # update learning rate

        lr = self.cosine_annealing(epoch)

        backend.set_value(self.model.optimizer.lr, lr)

        # log value

        self.lrs.append(lr)



    # when epoch ends check if there is a need to save a snapshot

    def on_epoch_end(self, epoch, logs={}):

        if (epoch + 1) % self.n_epochs_per_model == 0:

            # save model to file

            filename = self.__snapshot_name_fmt % ((epoch + 1) // self.n_epochs_per_model)

            self.model.save(filename)

            if self.verbose:

                print('Epoch %d: snapshot saved to %s' % (epoch, filename))

                

    # load all snapshots after training

    def load_ensemble(self):

        models = []

        for i in range(self.n_models):

            models.append(load_model(self.__snapshot_name_fmt % (i + 1)))

        return models

import pandas as pd

import os



# read train and test csvs

path = '/kaggle/input/digit-recognizer/'

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')



# extract target and remove from train

target = train['label']

train.drop(columns=['label'], inplace=True)



# reshape datasets according to image size (with 1 channel)

im_size = 28

train = train.to_numpy().reshape((-1, im_size, im_size, 1))

test = test.to_numpy().reshape((-1, im_size, im_size, 1))



# adjust pixels brightnesses to range 0..1

train = train / 255

test = test / 255



train.shape, test.shape
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



target = to_categorical(target)



x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=289)



x_train.shape, x_test.shape
plt.figure(1, figsize=(14, 10))

for i in range(18):

    plt.subplot(3, 6, i + 1)

    plt.imshow(x_train[i].reshape((im_size, im_size)), cmap='gray')

    plt.title(str(np.argmax(y_train[i])))

plt.show()
from keras.models import Sequential

from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense



model = Sequential()

model.add(Conv2D(16, 3, activation='relu', padding='same', input_shape=(im_size, im_size, 1)))

model.add(Dropout(0.5))

model.add(Conv2D(16, 3, activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(Conv2D(16, 5, activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Conv2D(32, 3, activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(Conv2D(32, 3, activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(Conv2D(32, 5, activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Conv2D(64, 3, activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(Conv2D(64, 3, activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(Conv2D(64, 5, activation='relu', padding='same'))

model.add(MaxPooling2D())



model.add(Flatten())



model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.summary()



model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['acc']

)
from keras.preprocessing.image import ImageDataGenerator



imagegen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2

)
se_callback = SnapshotEnsemble(n_models=7, n_epochs_per_model=50, lr_max=.001)



history = model.fit_generator(

    imagegen.flow(x_train, y_train, batch_size=32),

    steps_per_epoch=len(x_train) / 32,

    epochs=se_callback.n_epochs_total,

    verbose=0,

    callbacks=[se_callback],

    validation_data=(x_test, y_test)

)
h = history.history

plt.figure(1, figsize=(16, 10))



plt.subplot(121)

plt.xlabel('epoch')

plt.ylabel('loss')

plt.plot(h['loss'], label='training')

plt.plot(h['val_loss'], label='validation')

plt.legend()



plt.subplot(122)

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.plot(h['acc'], label='training')

plt.plot(h['val_acc'], label='validation')

plt.legend()



plt.show()
from sklearn.metrics import accuracy_score



# makes prediction according to given models and given weights

def predict(models, data, weights=None):

    if weights is None:

        # default weights provide voting equality

        weights = [1 / (len(models))] * len(models)

    pred = np.zeros((data.shape[0], 10))

    for i, model in enumerate(models):

        pred += model.predict(data) * weights[i]

    return pred

    

# returns accuracy for given predictions

def evaluate(preds, weights=None):

    if weights is None:

        weights = [1 / len(preds)] * len(preds)

    y_pred = np.zeros((y_test.shape[0], 10))

    for i, pred in enumerate(preds):

        y_pred += pred * weights[i]

    y_pred = np.argmax(y_pred, axis=1)

    y_true = np.argmax(y_test, axis=1)

    return accuracy_score(y_true, y_pred)



# load list of snapshots

models = se_callback.load_ensemble()

# precalculated predictions of all models

preds = []

# evaluate every model as single

for i, model in enumerate(models):

    pred = predict([model], x_test)

    preds.append(pred)

    score = evaluate([pred])

    print(f'model {i + 1}: accuracy = {score:.4f}')



# evaluate ensemble (with voting equality)

ensemble_score = evaluate(preds)

print(f'ensemble: accuracy = {ensemble_score:.4f}')
best_score = ensemble_score

best_weights = None

no_improvements = 0

while no_improvements < 5000: #patience

    

    # generate normalized weights

    new_weights = np.random.uniform(size=(len(models), ))

    new_weights /= new_weights.sum()

    

    # get the score without predicting again

    new_score = evaluate(preds, new_weights)

    

    # check (and save)

    if new_score > best_score:

        no_improvements = 0

        best_score = new_score

        best_weights = new_weights

        print(f'improvement: {best_score:.4f}')

    else:

        no_improvements += 1



print(f'best weights are {best_weights}')
pred = predict(models, test, best_weights)



res = pd.DataFrame()

res['ImageId'] = np.arange(test.shape[0]) + 1

res['Label'] = np.argmax(pred, axis=1)

res.to_csv('submission.csv', index=False)

res.head(15)