import numpy as np

import pandas as pd

import os

import math

from tqdm.notebook import tqdm

from sklearn.exceptions import ConvergenceWarning

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=ConvergenceWarning)



PATH = '/kaggle/input/cat-in-the-dat-ii/'

train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'test.csv')



# separate target, remove id and target

test_ids = test['id']

target = train['target']

train.drop(columns=['id', 'target'], inplace=True)

test.drop(columns=['id'], inplace=True)



train.head()
import category_encoders as ce



te = ce.TargetEncoder(cols=train.columns.values, smoothing=0.3).fit(train, target)



train = te.transform(train)

train.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=289)



x_train.shape, x_test.shape
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
from keras.models import Sequential

from keras.layers import Dense, Dropout, LeakyReLU



model = Sequential()

model.add(Dense(32, input_shape=(train.shape[1], )))

model.add(LeakyReLU())

model.add(Dropout(0.5))

model.add(Dense(16))

model.add(LeakyReLU())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.summary()



model.compile(

    loss='binary_crossentropy',

    optimizer='adam',

    metrics=['acc']

)
se_callback = SnapshotEnsemble(n_models=7, n_epochs_per_model=15, lr_max=.01)



history = model.fit(

    x_train,

    y_train,

    epochs=se_callback.n_epochs_total,

    verbose=1,

    batch_size=32,

    callbacks=[se_callback],

    validation_data=(x_test, y_test)

)
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

    pred = np.zeros((data.shape[0], ))

    for i, model in enumerate(models):

        pred += model.predict(data).flatten() * weights[i]

    return pred



# returns roc auc for preds and weights

def evaluate(preds, weights=None):

    if weights is None:

        weights = [1 / len(preds)] * len(preds)

    y_true = np.zeros((y_test.shape[0], ))

    for i, pred in enumerate(preds):

        y_true += pred.flatten() * weights[i]

    return roc_auc_score(y_test, y_true)



# load list of snapshots

models = se_callback.load_ensemble()

preds = []

# evaluate every model as single

for i, model in enumerate(models):

    pred = predict([model], x_test)

    preds.append(pred)

    score = evaluate([pred])

    print(f'model {i + 1}: roc = {score:.4f}')



# evaluate ensemble (with voting equality)

ensemble_score = evaluate(preds)

print(f'ensemble: roc = {ensemble_score:.4f}')
best_score = ensemble_score

best_weights = None

no_improvements = 0

while no_improvements < 500: #patience

    

    # generate normalized weights

    new_weights = np.random.uniform(size=(len(models), ))

    new_weights /= new_weights.sum()

    

    # get the score (no extra predictions)

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
# transform test and predict

test = te.transform(test)

pred = predict(models, test, best_weights)



res = pd.DataFrame()

res['id'] = test_ids

res['target'] = pred

res.to_csv('submission.csv', index=False)

res.head(15)