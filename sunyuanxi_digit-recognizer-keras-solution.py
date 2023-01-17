import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from sklearn.preprocessing import label_binarize

from sklearn.svm import SVC

from itertools import cycle

from scipy import interp



from keras.optimizers import Adam

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Input, Dropout, Flatten

from keras.models import Model, load_model



from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['label'], axis=1), train['label'], random_state = 0)
def plot_vector(vec):

    '''

    Takes in image vector, transforms and plots

    '''

    v_sq = vec.values.reshape((28,28))

    plt.imshow(v_sq, interpolation='nearest', cmap = 'gray')
fig = plt.figure(figsize=(8, 8))

for i in range(16):

    fig.add_subplot(4, 4, i + 1)

    plot_vector(X_train.iloc[i])

    plt.title(str(y_train.iloc[i]))

    plt.xticks([])

    plt.yticks([])
X_train = X_train.values.reshape(-1,28,28,1)

X_valid = X_valid.values.reshape(-1,28,28,1)

y_train = label_binarize(y_train, classes=range(10))

y_valid = label_binarize(y_valid, classes=range(10))
class RocAucEvaluation(Callback):



    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
class CNN:



    def __init__(self):

        self.arguments  = {

                    'batch_size': 64,

                    'epochs': 100,

                    'learning_rate': 1e-3,

                    'learning_rate_decay': 0,

                    'units': 128,

                    'drop_out_rate': 0.2,

                    'checkpoint_path': 'best_bilstm_model.hdf5',

                    'early_stop_patience': 10,

                }

        print('Building CNN Models ...')

        print(self.arguments)





    def fit(self, X_train, y_train, X_valid, y_valid):



        file_path = self.arguments['checkpoint_path']

        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                      save_best_only = True, mode = "min")

        ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = self.arguments['early_stop_patience'])



        inp = Input(shape=(28,28,1))

        x = Conv2D(filters=20, kernel_size = (5, 5), activation="relu")(inp)

        max_pool_x = MaxPooling2D(pool_size=(2,2))(x)

        y = Conv2D(filters=20, kernel_size = (5, 5), activation="relu")(max_pool_x)

        max_pool_y = MaxPooling2D(pool_size=(2,2))(y)

        flat = Flatten()(max_pool_y)

        z = Dense(100, activation="relu")(flat)

        z = Dropout(rate = self.arguments['drop_out_rate'])(z)

        z = Dense(100, activation="relu")(z)

        z = Dropout(rate = self.arguments['drop_out_rate'])(z)

        output = Dense(10, activation="softmax")(z)



        self.model = Model(inputs = inp, outputs = output)

        self.model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = self.arguments['learning_rate'],

                           decay = self.arguments['learning_rate_decay']), metrics = ["accuracy"])

        history = self.model.fit(X_train, y_train, batch_size = self.arguments['batch_size'], epochs = self.arguments['epochs'],

                                 validation_data = (X_valid, y_valid), verbose = 1, callbacks = [ra_val, check_point, early_stop])

        self.model = load_model(file_path)



        print('Finished Building CNN Model as class attribute class.model')

        return self





    def predict(self, X, batch_size,  verbose):

        return self.model.predict(X, batch_size = batch_size, verbose = verbose)
cnn = CNN()

cnn.fit(X_train, y_train, X_valid, y_valid)

cnn_pred = cnn.predict(X_valid, batch_size = cnn.arguments['batch_size'], verbose = 1)
cnn.model.summary()
datagen = ImageDataGenerator(

          rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

          zoom_range = 0.1, # Randomly zoom image 

          width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

          height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

          )
train_gen = datagen.flow(X_train, y_train, batch_size=64)

valid_gen = datagen.flow(X_valid, y_valid, batch_size=64)
def view_aug(x, y, n=6):

    plt.subplots(1, n+1, figsize=(16,4))

    for i in range(n+1):

        aug = x if i==0 else next(datagen.flow(np.array([x]), np.array([y]), batch_size=4))[0]

        plt.subplot(1, n+1, i+1)

        plt.imshow(aug.reshape(28, 28), cmap = 'gray')

        plt.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)

        plt.title("Original %d"%y.argmax() if i==0 else "Aug #%s"%i)

    plt.show()



view_aug(X_train[1], y_train[1])
class CNN_Generator:



    def __init__(self):

        self.arguments  = {

                    'batch_size': 64,

                    'epochs': 100,

                    'learning_rate': 1e-3,

                    'learning_rate_decay': 0,

                    'units': 128,

                    'drop_out_rate': 0.2,

                    'checkpoint_path': 'best_bilstm_model.hdf5',

                    'early_stop_patience': 10,

                }

        print('Building CNN_Generator Models ...')

        print(self.arguments)





    def fit(self, train_gen, valid_gen):



        file_path = self.arguments['checkpoint_path']

        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                      save_best_only = True, mode = "min")

        ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = self.arguments['early_stop_patience'])



        inp = Input(shape=(28,28,1))

        x = Conv2D(filters=20, kernel_size = (5, 5), activation="relu")(inp)

        max_pool_x = MaxPooling2D(pool_size=(2,2))(x)

        y = Conv2D(filters=20, kernel_size = (5, 5), activation="relu")(max_pool_x)

        max_pool_y = MaxPooling2D(pool_size=(2,2))(y)

        flat = Flatten()(max_pool_y)

        z = Dense(100, activation="relu")(flat)

        z = Dropout(rate = self.arguments['drop_out_rate'])(z)

        z = Dense(100, activation="relu")(z)

        z = Dropout(rate = self.arguments['drop_out_rate'])(z)

        output = Dense(10, activation="softmax")(z)



        self.model = Model(inputs = inp, outputs = output)

        self.model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = self.arguments['learning_rate'],

                           decay = self.arguments['learning_rate_decay']), metrics = ["accuracy"])

        history = self.model.fit_generator(train_gen, epochs = self.arguments['epochs'],

                                           steps_per_epoch = X_train.shape[0] // self.arguments['batch_size'],

                                           validation_steps = X_train.shape[0] // self.arguments['batch_size'],

                                           validation_data = valid_gen, verbose = 1, callbacks = [ra_val, check_point, early_stop])

        self.model = load_model(file_path)



        print('Finished Building CNN Model as class attribute class.model')

        return self





    def predict(self, X, batch_size,  verbose):

        return self.model.predict(X, batch_size = batch_size, verbose = verbose)
cnn_gen = CNN_Generator()

cnn_gen.fit(train_gen, valid_gen)

cnn_gen_pred = cnn_gen.predict(X_valid, batch_size = cnn.arguments['batch_size'], verbose = 1)
train_predictions = []

valid_predictions = []

test_predictions = []

for i in range(5):

    cnn = CNN_Generator()

    cnn.fit(train_gen, valid_gen)

    train_predictions += [cnn.predict(X_train, batch_size = cnn.arguments['batch_size'], verbose = 0)]

    valid_predictions += [cnn.predict(X_valid, batch_size = cnn.arguments['batch_size'], verbose = 1)]

    test_predictions += [cnn.predict(test.values.reshape(-1,28,28,1), batch_size = cnn.arguments['batch_size'], verbose = 1)]
train_pred = np.concatenate(train_predictions, axis=1)

valid_pred = np.concatenate(valid_predictions, axis=1)

test_pred = np.concatenate(test_predictions, axis=1)

svm = SVC(probability=True, gamma='scale').fit(train_pred, np.argmax(y_train, axis=1))

pred = svm.predict(valid_pred)

print("val_acc: ", round(np.sum(pred == np.argmax(y_valid, axis=1))/y_valid.shape[0], 4))

print("val_ROC-AUC: ", round(roc_auc_score(y_valid, svm.predict_proba(valid_pred)), 6))
cm = confusion_matrix(np.argmax(y_valid, axis=1), pred)

num_classes = cm.shape[0]

count = np.unique(np.argmax(y_valid, axis=1), return_counts=True)[1].reshape(num_classes, 1)



fig = plt.figure(figsize=(10,6))

ax = plt.subplot(111)

im = ax.imshow(cm/count, cmap='YlGnBu')

im.set_clim(0, 1)

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(num_classes))

ax.set_yticks(np.arange(num_classes))

plt.yticks(fontsize=13)

plt.xticks(fontsize=13)

for i in range(num_classes):

    for j in range(num_classes):

        text = ax.text(i, j, cm[j][i], ha="center", va="center", color="w" if (cm/count)[j, i] > 0.5 else "black", fontsize=13)

ax.set_ylabel('True Label', fontsize=16)

ax.set_xlabel('Predicted Label', fontsize=16)

ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')

plt.show()
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission['Label'] = svm.predict(test_pred)

sample_submission.head()

sample_submission.to_csv('submission.csv', index=False)