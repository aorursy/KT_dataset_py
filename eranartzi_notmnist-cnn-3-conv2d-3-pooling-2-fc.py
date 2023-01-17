import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
dir_ = '../input/notMNIST_small/notMNIST_small/'
letters = os.listdir(dir_)
# Retrieve pictures files names
pictures_files = {}
for letter in letters:
    images = [name for name in os.listdir(dir_ + '%s/' % letter) if name[-4:] == '.png']
    pictures_files[letter] = images
# Get the actual pictures
data = {}
for letter in letters:
    print('---------------------------')
    print('Retrieving for %s' % letter)
    print('---------------------------')
    images = []
    for name in pictures_files[letter]:
        try:
            images.append(plt.imread(dir_+'{}/{}'.format(letter, name)))
        except Exception as e:
            print(e, name)
    data[letter] = images
print('Done')
from sklearn.preprocessing import  LabelEncoder
# Merge all data to one list
X = []
Y = []
X_nd = np.zeros(shape=(18724, 28, 28))
Y_nd = np.zeros(shape=(18724))
for key, list_ in data.items():
    for img in list_:
        X.append(img)
        Y.append(key)

for i in range(len(X)):
    X_nd[i, :, :] = X[i]
    

lbl_enc = LabelEncoder()
labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
lbl_enc.fit(labels)
Y = lbl_enc.transform(Y)
Y_nd = keras.utils.np_utils.to_categorical(Y, num_classes=10)
X_nd = np.expand_dims(X, -1).astype('float32')/255.0
from sklearn.model_selection import train_test_split
X_train, X_dev, Y_train, Y_dev = train_test_split(X_nd, Y_nd, test_size=.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=.1)
print('Training size: %s' % len(X_train))
print('DevSet size: %s' % len(X_dev))
print('TestSet size: %s' % (len(X_test)))
len(X_train) == len(Y_train)
print(Y_train.shape)
print(X_train.shape)
import warnings

class EarlyStoppingOverFit(keras.callbacks.EarlyStopping):
    
    
    def __init__(self, monitor='overfit', min_delta=0, patience=0, verbose=0, mode='max'):
        super(EarlyStoppingOverFit, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        
        
    def on_epoch_end(self, epoch, logs=None):
        if self.monitor == 'overfit':
            current = logs.get('acc') - logs.get('val_acc')
        else:
            current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        self.monitor_op = np.greater
        if self.monitor_op(self.min_delta, np.linalg.norm(current)):
            print('No overfitting with %s' % np.linalg.norm(current))
            self.best = current
            self.wait = 0
        elif logs['acc'] > logs['val_acc']:
            print('Overfitting with %s' % np.linalg.norm(current))
            self.wait += 1
#             print('Overfitting detected for %s epochs' % self.wait)
            if self.wait >= self.patience:
                print('Stopping for overfitting')
                self.stopped_epoch = epoch
                self.model.stop_training = True
## Now build the model
inputs = keras.layers.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu', padding='same')(inputs)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(rate=.3)(x)
x = keras.layers.Dense(units=128)(x)
x = keras.layers.Dropout(rate=.3)(x)
outputs = keras.layers.Dense(units=10, activation='softmax')(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# Callback
over_fit = EarlyStoppingOverFit(min_delta=0.015, patience=2)
early_stop = keras.callbacks.EarlyStopping(min_delta=0.01, monitor='val_acc', patience=10, verbose=1)
lr_decay = keras.callbacks.ReduceLROnPlateau()
model.build(None,)
#Train
model.fit(X_train,
          Y_train,
          validation_data=(X_dev, Y_dev),
          epochs=50,
          batch_size=64,
          verbose=1,
          callbacks=[over_fit, early_stop, lr_decay])
history = model.history.history
history.keys()
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(30, 7))
epochs = range(len(history['acc']))
ax0.set_title('Accuracy')
ax0.plot(epochs, history['acc'], '-b')
ax0.plot(epochs, history['val_acc'], '-g')

ax1.set_title('Loss')
ax1.plot(epochs, history['loss'], '-b')
ax1.plot(epochs, history['val_loss'], '-g')
# lets see a few examples we got wrong
preds = model.predict(X_test)
# Get index of max value per data point
preds_idx = np.argmax(preds, axis=-1)
y_true_idx = np.argmax(Y_test, axis=-1)
preds_labels = lbl_enc.inverse_transform(preds_idx)
y_true_labels = lbl_enc.inverse_transform(y_true_idx)
join = np.stack([preds_labels, y_true_labels], axis=1)
df = pd.DataFrame(join, columns=['pred', 'true'])
sample_of_wrongs = df[df.pred != df.true].head(9)
sample_of_wrongs
x_test_reshape = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
f, axs = plt.subplots(9, figsize=(20,20))
print(type(axs[0]))
graphs_imgs = zip(axs, sample_of_wrongs.index.values)
for ax, i in graphs_imgs:
    ax.imshow(x_test_reshape[i, :, :], cmap='gray')