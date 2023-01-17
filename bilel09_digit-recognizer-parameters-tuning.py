import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorboard.plugins.hparams import api as hp
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

tf.config.experimental.list_physical_devices('GPU')
train = pd.read_csv('../input/digit-recognizer/train.csv')
train.head()
Y_train = train['label'].to_numpy()
X_train = train.iloc[:, 1:].to_numpy()
del train
X_train = X_train.reshape(-1,28,28,1)
sns.countplot(Y_train)
plt.show()
X_test = pd.read_csv('../input/digit-recognizer/test.csv').to_numpy()
X_test = X_test.reshape(-1,28,28,1)


X_train = X_train / 255.0
X_test = X_test / 255.0
from keras.utils.np_utils import to_categorical 
Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=42)
io.imshow(X_train[0][:,:,0])
batch_size = 32
epochs = 10
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)


datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        )

%load_ext tensorboard
!rm -rf ./logs/ 
HP_NUM_FILTERS = hp.HParam('num_filters', hp.Discrete([32, 64]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 512]))
HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3, 5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_FILTERS, HP_NUM_UNITS, HP_OPTIMIZER, HP_KERNEL_SIZE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_val_model(logdir, hparams):
    batch_size = 32
    model = tf.keras.models.Sequential([
        Conv2D(hparams[HP_NUM_FILTERS], hparams[HP_KERNEL_SIZE], padding='same', activation='relu', input_shape=(28, 28, 1)),
        Conv2D(hparams[HP_NUM_FILTERS], hparams[HP_KERNEL_SIZE], padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(hparams[HP_NUM_FILTERS]*2, hparams[HP_KERNEL_SIZE], padding='same', activation='relu'),
        Conv2D(hparams[HP_NUM_FILTERS]*2, hparams[HP_KERNEL_SIZE], padding='same', activation='relu'),
        MaxPooling2D(strides=(2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(hparams[HP_NUM_UNITS], activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit( datagen.flow( X_train,Y_train, batch_size=batch_size), verbose=0,
                            epochs = 10, validation_data = (X_val,Y_val),
                            steps_per_epoch=X_train.shape[0] // batch_size
                            , callbacks=[learning_rate_reduction,
                             tf.keras.callbacks.TensorBoard(logdir)
                                        ]
            ) 
            
    accuracy = np.array(history.history['val_accuracy']).max()
    print(f'best accuracy: {round(accuracy,4)}')
    return accuracy
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_val_model(run_dir, hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    return accuracy 
from itertools import product

session_num = 0
results = []
for num_filters, num_units, kernel_size, optimizer in product(HP_NUM_FILTERS.domain.values, 
                                                HP_NUM_UNITS.domain.values,
                                                HP_KERNEL_SIZE.domain.values,
                                                HP_OPTIMIZER.domain.values): 
    hparams = {
        HP_NUM_FILTERS: num_filters,
        HP_NUM_UNITS: num_units,
        HP_KERNEL_SIZE: kernel_size,
        HP_OPTIMIZER: optimizer,
    }

    run_name = f'run{session_num}_{str({h.name: hparams[h] for h in hparams})}' 
    print(f'---{run_name}')
    accuracy = run('logs/hparam_tuning/' + run_name, hparams)
    session_num += 1
    
    results.append([num_filters, num_units, kernel_size, optimizer, accuracy])

results = pd.DataFrame(results, columns=['num_filters', 'num_units', 'kernel_size', 'optimizer', 'accuracy'])

results.sort_values(by='accuracy', ascending=False).head()

fig, axs = plt.subplots(ncols=4, figsize=(20,5), sharey=True)
for i, col in enumerate(results.columns[0:-1]):
    sns.scatterplot(x=col, y='accuracy', data=results, ax=axs[i])
%tensorboard --logdir logs/hparam_tuning
best_params = results.iloc[results['accuracy'].argmax()]
best_params
model = tf.keras.models.Sequential([
        Conv2D(best_params['num_filters'], int(best_params['kernel_size']), padding='same', activation='relu', input_shape=(28, 28, 1)),
        Conv2D(best_params['num_filters'], int(best_params['kernel_size']), padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(best_params['num_filters']*2, int(best_params['kernel_size']), padding='same', activation='relu'),
        Conv2D(best_params['num_filters']*2, int(best_params['kernel_size']), padding='same', activation='relu'),
        MaxPooling2D(strides=(2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(best_params['num_units'], activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

model.compile(
        optimizer=best_params['optimizer'],
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 20, validation_data = (X_val,Y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction, early_stopping])
def plot_fit(history):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].plot(history.history['loss'], label='Train loss')
    axes[0].plot(history.history['val_loss'], label='Validation loss')
    axes[1].plot(history.history['accuracy'], label='Train accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation accuracy')
    axes[0].legend()
    axes[1].legend()
    plt.show()


plot_fit(history)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
val_predictions = model.predict(X_val)
cm = confusion_matrix(np.argmax(Y_val, axis=1), np.argmax(val_predictions, axis=1), normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm.round(2), display_labels=np.arange(10))

disp = disp.plot()

plt.show()
def plot_image(predictions_array, img, true_prediction):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)

  plt.xlabel(f"Pred: {predicted_label} ({100*np.max(predictions_array):2.0f}%) | True: {true_prediction}")

def plot_value_array(predictions_array):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')



wrong_predictions = np.argwhere(np.argmax(Y_val, axis=1) !=np.argmax(val_predictions, axis=1))

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i, image_idx in enumerate(wrong_predictions[0:num_images].reshape(-1)):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(val_predictions[image_idx], X_val[image_idx].reshape(28,28), np.argmax(Y_val[image_idx]))
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(val_predictions[image_idx])
plt.tight_layout()
plt.show()

model = tf.keras.models.Sequential([
        Conv2D(best_params['num_filters'], int(best_params['kernel_size']), padding='same', activation='relu', input_shape=(28, 28, 1)),
        Conv2D(best_params['num_filters'], int(best_params['kernel_size']), padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(best_params['num_filters']*2, int(best_params['kernel_size']), padding='same', activation='relu'),
        Conv2D(best_params['num_filters']*2, int(best_params['kernel_size']), padding='same', activation='relu'),
        MaxPooling2D(strides=(2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(best_params['num_units'], activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

model.compile(
        optimizer=best_params['optimizer'],
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
X_train = np.concatenate((X_train,X_val))
Y_train = np.concatenate((Y_train,Y_val))

model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 14,
                              steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis=1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)