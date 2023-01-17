import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



PATH = '/kaggle/input/plant-pathology-2020-fgvc7/'



train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'test.csv')



target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]

test_ids = test['image_id']



train_len = train.shape[0]

test_len = test.shape[0]



train.describe()
from PIL import Image

from tqdm.notebook import tqdm



SIZE = 224



train_images = np.empty((train_len, SIZE, SIZE, 3))

for i in tqdm(range(train_len)):

    train_images[i] = np.uint8(Image.open(PATH + f'images/Train_{i}.jpg').resize((SIZE, SIZE)))

    

test_images = np.empty((test_len, SIZE, SIZE, 3))

for i in tqdm(range(test_len)):

    test_images[i] = np.uint8(Image.open(PATH + f'images/Test_{i}.jpg').resize((SIZE, SIZE)))



train_images.shape, test_images.shape
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train_images, target.to_numpy(), test_size=0.2, random_state=289) 



x_train.shape, x_test.shape, y_train.shape, y_test.shape
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=289)



x_train, y_train = ros.fit_resample(x_train.reshape((-1, SIZE * SIZE * 3)), y_train)

x_train = x_train.reshape((-1, SIZE, SIZE, 3))

x_train.shape, y_train.sum(axis=0)
import gc



del train_images

gc.collect()
from keras.callbacks import Callback

from keras import backend

from keras.models import load_model

import math



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
from keras.models import Model, Sequential, load_model, Input

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.utils import plot_model

from keras.regularizers import l2



filters = 32

reg = .0005



model = Sequential()



for i in range(5):

    model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg), input_shape=(SIZE, SIZE, 3)))

    model.add(LeakyReLU())

    

    model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg)))

    model.add(LeakyReLU())

    

    if i != 4:

        model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))

        model.add(LeakyReLU())

        

    model.add(MaxPooling2D())

    model.add(Dropout(0.5))

    model.add(BatchNormalization())



    filters *= 2



model.add(Flatten())

model.add(Dense(4, activation='softmax'))



model.summary()



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['acc']

)
from keras.preprocessing.image import ImageDataGenerator



imagegen = ImageDataGenerator(

    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True

)



se_callback = SnapshotEnsemble(n_models=3, n_epochs_per_model=300, lr_max=.005)



history = model.fit_generator(

    imagegen.flow(x_train, y_train, batch_size=32),

    epochs=se_callback.n_epochs_total,

    steps_per_epoch=x_train.shape[0] // 32,

    verbose=1,

    callbacks=[se_callback],

    validation_data=(x_test, y_test)

)



# load list of snapshots

models = se_callback.load_ensemble()
from matplotlib import pyplot as plt

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
from sklearn.metrics import roc_auc_score



# makes prediction according to given models and given weights

def predict(models, data, weights=None):

    if weights is None:

        # default weights provide voting equality

        weights = [1 / (len(models))] * len(models)

    pred = np.zeros((data.shape[0], 4))

    for i, model in enumerate(models):

        pred += model.predict(data) * weights[i]

    return pred

    

# returns roc auc for given predictions

def evaluate(preds, weights=None):

    if weights is None:

        weights = [1 / len(preds)] * len(preds)

    y_pred = np.zeros((y_test.shape[0], 4))

    for i, pred in enumerate(preds):

        y_pred += pred * weights[i]

    return roc_auc_score(y_test, y_pred)



# load list of snapshots

models = se_callback.load_ensemble()

# precalculated predictions of all models

preds = []

# evaluate every model as single

for i, model in enumerate(models):

    pred = predict([model], x_test)

    preds.append(pred)

    score = evaluate([pred])

    print(f'model {i + 1}: roc auc = {score:.4f}')



# evaluate ensemble (with voting equality)

ensemble_score = evaluate(preds)

print(f'ensemble: roc auc = {ensemble_score:.4f}')
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
pred = predict(models, test_images, best_weights)



res = pd.DataFrame()

res['image_id'] = test_ids

res['healthy'] = pred[:, 0]

res['multiple_diseases'] = pred[:, 1]

res['rust'] = pred[:, 2]

res['scab'] = pred[:, 3]

res.to_csv('submission.csv', index=False)

res.head(40)