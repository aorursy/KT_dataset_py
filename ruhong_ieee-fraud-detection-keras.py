# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, regularizers

from sklearn.metrics import roc_auc_score

import keras.backend as K

from collections import OrderedDict

from keras.models import Model

import seaborn as sns



print(f'tf={tf.__version__}, keras={keras.__version__}')
SEED = 31

EPOCHS = 100

BATCH_SIZE = 128

TARGET = 'isFraud'

VALIDATION_PERCENT = 0.08

DROPOUT_RATE = 0.5  #0.2

L2 = 0.0001  #0.00001

LEARNING_RATE = 0.002

HIDDEN_UNITS = 200  #256

HIDDEN_LAYERS = 2  #4

PATIENCE = 4

DECAY_RATE = 0.5

DECAY_STEPS = 4
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.set_random_seed(seed)





seed_everything(SEED)
file_folder = '../input/ieee-fraud-detection-preprocess'

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

train.info()
test.info()
def group_ratios_and_pcs():

    excludes = {TARGET}

    for i in range(1, 340):

        excludes.add(f'V{i}')

    cols = set(train.columns.values) - excludes

    return list(cols)





def _keep(col):

    if col == TARGET:

        return False

    if col.startswith('_pc_'):

        return False

    #if '_to_' in col:

     #   return False

    return True





PREDICTORS = [c for c in train.columns.values if _keep(c)]

#PREDICTORS = group_ratios_and_pcs()

print(f'{len(PREDICTORS)} predictors={PREDICTORS}')
val_size = int(VALIDATION_PERCENT * len(train))

train, val = train[:-val_size], train[-val_size:]

print(f'train={train.shape}, val={val.shape}')
y_train = train[TARGET]

x_train = train[PREDICTORS]

y_val = val[TARGET]

x_val = val[PREDICTORS]

x_test = test[PREDICTORS]
class AucRocCallback(keras.callbacks.Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]





    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)[:,1]

        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)[:,1]

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('epoch=%d, roc-auc: val=%s, train=%s' % (epoch,str(round(roc_val,4)),str(round(roc,4))))

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
def build_model(input_shape):

  #ini = keras.initializers.he_uniform()

  ini = keras.initializers.he_normal()

  reg = regularizers.l2(L2)  



  def _block():

    res = []

    for i in range(HIDDEN_LAYERS):

        if i == 0:

            res.append(layers.Dense(HIDDEN_UNITS,

                                    activation=tf.nn.relu,

                                    kernel_initializer=ini,

                                    kernel_regularizer=reg,

                                    input_shape=input_shape,

                                    name=f'dense_{i}'))

        else:

            res.append(layers.Dense(HIDDEN_UNITS,

                                    activation=tf.nn.relu,

                                    kernel_initializer=ini,

                                    kernel_regularizer=reg,

                                    name=f'dense_{i}'))

        res.append(layers.BatchNormalization(name=f'batch_norm_{i}'))

        res.append(layers.Dropout(DROPOUT_RATE, name=f'dropout_{i}'))

    return res

  

  hls = _block()

  model = keras.Sequential(hls + [

    layers.Dense(2, activation=tf.nn.softmax, name='output')

  ])

  opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

  model.compile(optimizer=opt,

              loss='sparse_categorical_crossentropy',

              metrics=['acc'])

  return model





model = build_model(input_shape=[len(x_train.keys())])

model.summary()
# sanity check model is producing output of desired type and shape

example_batch = x_train[:10]

example_result = model.predict(example_batch)

example_result
def calc_stats(W):

    return np.linalg.norm(W, 2), np.mean(W), np.std(W)





class MyDebugWeights(keras.callbacks.Callback):

    

    def __init__(self):

        super(MyDebugWeights, self).__init__()

        self.weights = []

        self.tf_session = K.get_session()

            

    def on_epoch_end(self, epoch, logs=None):

        for layer in self.model.layers:

            name = layer.name

            for i, w in enumerate(layer.weights):

                w_value = w.eval(session=self.tf_session)

                w_norm, w_mean, w_std = calc_stats(np.reshape(w_value, -1))

                self.weights.append((epoch, "{:s}/W_{:d}".format(name, i), 

                                     w_norm, w_mean, w_std))

    

    def on_train_end(self, logs=None):

        for e, k, n, m, s in self.weights:

            print("{:3d} {:30s} {:7.3f} {:7.3f} {:7.3f}".format(e, k, n, m, s))

            





class WeightsLogger(keras.callbacks.Callback):

    

    def __init__(self):

        super(WeightsLogger, self).__init__()

        self.weights = pd.DataFrame(columns=['layer', 'target'])

        self.tf_session = K.get_session()

            

    def on_epoch_end(self, epoch, logs=None):

        for i, layer in enumerate(self.model.layers):

            name = layer.name

            if not name.startswith('dense_'):

                continue

            for w in layer.weights:

                w_value = w.eval(session=self.tf_session)

                tmp = pd.DataFrame(columns=['layer', 'target'])

                tmp['target'] = np.reshape(w_value, -1)

                tmp['layer'] = name

                self.weights = pd.concat([self.weights, tmp])

    



weights_logger = WeightsLogger()
def _learning_rate_scheduler(decay_rate, decay_steps):

    

    def _scheduler(epoch, lr):

        if (epoch + 1) % decay_steps == 0:

            return lr * decay_rate

        return lr



    return keras.callbacks.LearningRateScheduler(_scheduler, verbose=0)





lr_scheduler = _learning_rate_scheduler(decay_rate=DECAY_RATE, decay_steps=DECAY_STEPS)
%%time

# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)



auc_roc = AucRocCallback(training_data=(x_train, y_train), validation_data=(x_val, y_val))



def get_gradient_norm_func(model):

    grads = K.gradients(model.total_loss, model.trainable_weights)

    #summed_squares = [K.sum(K.square(g)) for g in grads]

    #norm = K.sqrt(sum(summed_squares))

    inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights

    func = K.function(inputs, [grads])

    return func





# must be placed right before fitting the model

get_gradient = get_gradient_norm_func(model)

history = model.fit(

  x_train, y_train,

  epochs=EPOCHS, verbose=0, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),

  callbacks=[auc_roc, early_stop, lr_scheduler, weights_logger])
model.trainable_weights
len(model.trainable_weights)
%%time

gs = get_gradient([x_train, y_train, np.ones(len(y_train))])[0]

print(f'len(gs)={len(gs)}')

gradients = pd.DataFrame(columns=['layer', 'target'])

node_names = [w.name for w in model.trainable_weights]

for i, node in enumerate(node_names):

    if node.startswith('dense_0'):

        layer = 'h0'

    elif node.startswith('dense_1'):

        layer = 'h1'

    elif node.startswith('dense_2'):

        layer = 'h2'

    elif node.startswith('dense_3'):

        layer = 'h3'

    elif node.startswith('dense_4'):

        layer = 'h4'

    else:

        continue

    tmp = pd.DataFrame(columns=['layer', 'target'])

    tmp['target'] = gs[i].flatten()

    tmp['layer'] = layer

    gradients = pd.concat([gradients, tmp])

    

    

gradients.info()
g_stats = gradients.groupby(['layer'])['target'].agg(['median', 'std', 'min', 'max', 'skew'])

g_stats.head(10)
g_ax = sns.violinplot(x='layer', y='target', data=gradients)

g_ax.set_title('Gradients')
weights = weights_logger.weights

weights.info()
w_stats = weights.groupby(['layer'])['target'].agg(['median', 'std', 'min', 'max', 'skew'])

w_stats.head(HIDDEN_LAYERS * 10)
w_ax = sns.violinplot(x='layer', y='target', data=weights)

w_ax.set_title('Weights')
_sample = train.sample(n=1000)

x_sample = _sample[PREDICTORS]

y_sample = _sample[TARGET]

activations = pd.DataFrame(columns=['layer', 'target'])

#layers = ['dense_0', 'dense_1']

layers = []

for i in range(HIDDEN_LAYERS):

    layers.append(f'batch_norm_{i}')



for i, layer in enumerate(layers):

    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer).output)

    intermediate_output = intermediate_layer_model.predict(x_sample)

    tmp = pd.DataFrame(columns=['layer', 'target'])

    tmp['target'] = [a for example in intermediate_output for a in example]

    tmp['layer'] = f'h{i}'

    activations = pd.concat([activations, tmp])

  

activations.info()
stats = activations.groupby(['layer'])['target'].agg(['median', 'std', 'min', 'max', 'skew'])

stats.head(len(layers))
ax = sns.violinplot(x='layer', y='target', data=activations)

ax.set_title('Activations')
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail(EPOCHS)
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Loss')

  plt.plot(hist['epoch'], hist['loss'],label='Train')

  plt.plot(hist['epoch'], hist['val_loss'],label = 'Val')

  #plt.ylim([0,5])

  plt.legend()

  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Accuracy')

  plt.plot(hist['epoch'], hist['acc'],label='Train')

  plt.plot(hist['epoch'], hist['val_acc'],label = 'Val')

  #plt.ylim([0,20])

  plt.legend()

  plt.figure()

  plt.xlabel('Learning Rate')

  plt.ylabel('Loss')

  plt.plot(hist['lr']*-1, hist['loss'],label='Train')

  plt.plot(hist['lr']*-1, hist['val_loss'],label = 'Val')

  plt.legend()

  plt.show()





plot_history(history)
x_test = test[PREDICTORS]

sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')

sub[TARGET] = model.predict(x_test)[:,1]

sub.head()
sub.to_csv('submission.csv', index=False)

print(os.listdir("."))