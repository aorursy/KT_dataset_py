import numpy as np

import sklearn as sk

import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd

import sklearn.model_selection as ms

import sklearn.preprocessing as p

import math
tf.version.VERSION
mnist = pd.read_csv('../input/digit-recognizer/train.csv')
mnist.shape, mnist.columns
height = 28

width = 28

channels = 1
n_outputs = 10
def show_digit(digit):

    plt.imshow(digit.reshape(height, width))

    plt.show()
def show_digit_and_print_label(row):

    print(row.loc['label'], ':')

    show_digit(row.loc[mnist.columns != 'label'].values)
mnist.loc[:3].apply(show_digit_and_print_label, axis=1)

X_data = mnist.drop(columns='label')

y_data = mnist['label']
y_data = tf.keras.utils.to_categorical(y_data, num_classes = n_outputs)

y_data.shape
X_train, X_val, y_train, y_val  = ms.train_test_split(X_data, y_data, test_size=0.15)
scaler = p.StandardScaler()

X_train = scaler.fit_transform(X_train)

X_train = X_train.reshape(-1, height, width, channels)



X_val = scaler.transform(X_val)

X_val = X_val.reshape(-1, height, width, channels)
X_train.shape, X_val.shape
y_train.shape, y_val.shape
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(

        rotation_range=10,

        zoom_range = 0.1, 

        width_shift_range=0.1, 

        height_shift_range=0.1

) 
batch_size = 250
train_data_gen = image_gen.flow(X_train, y=y_train, batch_size=batch_size)
class CosineAnnealingLearningRateCallback(tf.keras.callbacks.Callback):



    def __init__(self, n_epochs, n_cycles, lrate_max, n_epochs_for_saving, verbose=0):

        self.epochs = n_epochs

        self.cycles = n_cycles

        self.lr_max = lrate_max

        self.n_epochs_for_saving = n_epochs_for_saving

        self.best_val_acc_per_cycle = float('-inf')

        

    # allow to save model only in the last n_epochs_for_saving  

    def is_save_range(self, epoch, epochs_per_cycle, n_epochs_for_saving):

        epoch += 1



        f, d = math.modf(epoch / epochs_per_cycle)

        next_end = epochs_per_cycle * (d + (1 if f > 0 else 0))



        need_to_save = epoch > (next_end - n_epochs_for_saving) 

        return need_to_save



    # calculate learning rate for an epoch

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):

        epochs_per_cycle = math.floor(n_epochs/n_cycles)

        cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)

        return lrate_max/2 * (math.cos(cos_inner) + 1)

    

    # calculate and set learning rate at the start of the epoch

    def on_epoch_begin(self, epoch, logs=None):

        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        

    # make a snapshots if necessary

    def on_epoch_end(self, epoch, logs={}):

        # check if we can save model

        epochs_per_cycle = math.floor(self.epochs / self.cycles)

        

        if epoch % epochs_per_cycle == 0:

            self.best_val_acc_per_cycle = float('-inf')

            

            #test log

            #print('MyCheckpointer. New cycle - best_val_acc_per_cycle has been erased')

        

        isr = self.is_save_range(epoch, epochs_per_cycle, self.n_epochs_for_saving)

        

        last_val_acc = logs['val_accuracy']

        

        #test logs

        #print('MyCheckpointer. epoch: ', epoch)

        #print('MyCheckpointer. epochs_per_cycle: ', epochs_per_cycle)

        #print('MyCheckpointer. isr: ', isr)

        #print('MyCheckpointer. best_val_acc_per_cycle: ', self.best_val_acc_per_cycle, ', last_val_acc: ', last_val_acc)

        

        # check is snapshot necessary 

        if epoch != 0 and isr and last_val_acc > self.best_val_acc_per_cycle:

            self.best_val_acc_per_cycle = last_val_acc

            

            # save model to file

            filename = f'snapshot_model_{epoch // epochs_per_cycle}.h5'

            self.model.save(filename)

            print(f'saved snapshot {filename}, epoch: {epoch}, val_accuracy: {last_val_acc:.5f}')

        

# we can also play with these hyperparameters         

n_epochs = 300

n_cycles = n_epochs / 50

n_epochs_for_saving = 20



calrc = CosineAnnealingLearningRateCallback(n_epochs, n_cycles, 0.01, n_epochs_for_saving)
model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(32, 3, 1, padding='same', activation='relu', input_shape=(height, width, channels)))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Conv2D(32, 3, 1, padding='same', activation='relu', input_shape=(height, width, channels)))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.20))





model.add(tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.20))





model.add(tf.keras.layers.Conv2D(128, 3, 1, padding='same', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.20))



model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data_gen, batch_size=batch_size, epochs = n_epochs, validation_data = (X_val, y_val), callbacks=[calrc], verbose=2)
def load_all_models(n_models):

    all_models = list()

    for i in range(n_models):

        filename = f'snapshot_model_{str(i)}.h5'

        model = tf.keras.models.load_model(filename)

        all_models.append(model)

    return all_models

 

def ensemble_predictions(models, testX):

    yhats = [model.predict(testX) for model in models]

    yhats = np.array(yhats)

    summed = np.sum(yhats, axis=0)

    result = np.argmax(summed, axis=1)

    return result

 

def evaluate_n_models(models, n_models, testX, testy):

    subset = models[:n_models]

    yhat = ensemble_predictions(subset, testX)

    return sk.metrics.accuracy_score(testy, yhat)
models = load_all_models(6)



single_scores, ensemble_scores = list(), list()

for i in range(1, len(models)+1):



    ensemble_score = evaluate_n_models(models, i, X_val, np.argmax(y_val, axis=1))



    _, single_score = models[i-1].evaluate(X_val, y_val, verbose=0)



    print(f'{i}: single={single_score:.5f}, ensemble={ensemble_score:.5f}')

    ensemble_scores.append(ensemble_score)

    single_scores.append(single_score)



x_axis = [i for i in range(1, len(models)+1)]

plt.plot(x_axis, single_scores, marker='o', linestyle='None')

plt.plot(x_axis, ensemble_scores, marker='x')

plt.show()
X_pred = pd.read_csv('../input/digit-recognizer/test.csv')

X_pred = scaler.transform(X_pred)

X_pred = X_pred.reshape(-1, height, width, channels)
y_pred = pd.DataFrame()

y_pred['ImageId'] = pd.Series(range(1,X_pred.shape[0] + 1))

y_pred['Label'] = ensemble_predictions(models, X_pred)



y_pred.to_csv("submission.csv", index=False)