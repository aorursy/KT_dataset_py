import time
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow.keras.layers as l
import tensorflow.keras.optimizers as opt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json

np.random.RandomState(42)

data_path = '../input/Kannada-MNIST/'

train_path = data_path + 'train.csv'
test_path = data_path + 'test.csv'
sample_path = data_path + 'sample_submission.csv'

save_path = ''
load_path = '../input/kennada-mnist-pretrained-model/'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)

# convert dataframes to numpy matricies
X = train_df.drop('label', axis=1).to_numpy()
y = train_df['label'].to_numpy()
X_test = test_df.drop('id', axis=1).to_numpy()

# reshape X's for keras and encode y using one-hot-vector-encoding
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y)
X_test = X_test.reshape(-1, 28, 28, 1)

# normalize the data to range(0, 1)
X = X / 255
X_test = X_test / 255

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42) 

# split to train and validation sets

print('Train shape = {} {}'.format(X_train.shape, y_train.shape))
print('Valid shape = {} {}'.format(X_valid.shape, y_valid.shape))
print('Test shape = {}'.format(X_test.shape))

# model builder
def get_model():
    return Sequential([
        l.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),
        l.Conv2D(64,  (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),
        l.Conv2D(64,  (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),

        l.MaxPooling2D(2, 2),
        l.Dropout(0.25),

        l.Conv2D(128, (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),
        l.Conv2D(128, (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),
        l.Conv2D(128, (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),

        l.MaxPooling2D(2,2),
        l.Dropout(0.25),    

        l.Conv2D(256, (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
        l.LeakyReLU(alpha=0.1),
        l.Conv2D(256, (3,3), padding='same'),
        l.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),##
        l.LeakyReLU(alpha=0.1),

        l.MaxPooling2D(2,2),
        l.Dropout(0.25),

        l.Flatten(),
        l.Dense(256),
        l.LeakyReLU(alpha=0.1),

        l.BatchNormalization(),
        l.Dense(10, activation='softmax')
    ])

optimizers = {
    'sgd':          opt.SGD(),
    'sgd+momentum': opt.SGD(nesterov=True),
    'rmsprop':      opt.RMSprop(),
    'adam':         opt.Adam(),
}

batch_size = 1024
epochs = 5

datagen_train = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 0.1,
    zoom_range = 0.4,
    horizontal_flip = False
)

datagen_val = ImageDataGenerator()

learning_rate_reduction = ReduceLROnPlateau( 
    monitor='loss',
    factor=0.25,
    patience=2,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.00001
)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)

history = {}

for name, optimizer in optimizers.items():
    model = get_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    h = model.fit_generator(
        datagen_train.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train)//batch_size,
        epochs=epochs,
        validation_data=datagen_val.flow(X_valid, y_valid),
        validation_steps=50,
        callbacks=[learning_rate_reduction, es],
        verbose=1
    )

    history[name] = h.history

styles=[':','-.','--','-']
plt.figure(figsize=(20, 7))
    
for n, h in enumerate(history.values()):
    val_acc = h['val_accuracy']
    plt.plot(h['val_accuracy'], linestyle=styles[n])

plt.title('Model validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(optimizers.keys(), loc='upper left')

styles=[':','-.','--','-']
plt.figure(figsize=(20, 7))
    
for n, h in enumerate(history.values()):
    val_acc = h['accuracy']
    plt.plot(h['accuracy'], linestyle=styles[n])

plt.title('Model train accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(optimizers.keys(), loc='upper left')

styles=[':','-.','--','-']
plt.figure(figsize=(20, 7))
    
for n, h in enumerate(history.values()):
    val_acc = h['loss']
    plt.plot(h['loss'], linestyle=styles[n])

plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(optimizers.keys(), loc='upper left')

