# Imports and setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, SpatialDropout2D, BatchNormalization, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

%matplotlib inline

!pip install livelossplot --quiet
from livelossplot import PlotLossesKerasTF
# Load data
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
# Separate data into train and test
y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
X_test = test
print(X_train.shape, y_train.shape, X_test.shape)
# Normalize data
X_train /= 255
test /= 255
# Check normalization
X_train.iloc[0].agg(['min', 'max'])
# Reshape to numpy arrays, need third channel for keras
X_train = X_train.to_numpy().reshape(-1, 28, 28, 1)
X_test = test.to_numpy().reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# Train/Val split
random_seed = 8
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)
print(X_train.shape, X_val.shape)
# Visualize some images in training set
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    r = random.randint(0, X_train.shape[0] - 1)
    plt.title(y_train[r].argmax())
    plt.imshow(X_train[r][:, :, 0])
def model_builder():
    input_shape = (28, 28, 1)
    regularizer = regularizers.l2(5e-4)
    
    model = Sequential()
    model.add(Conv2D(32, 3, kernel_regularizer=regularizer, padding='same', activation='relu', input_shape=input_shape)) 
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(32, 3, kernel_regularizer=regularizer, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(32, 5, kernel_regularizer=regularizer, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(SpatialDropout2D(0.25))
    
    model.add(Conv2D(64, 3, kernel_regularizer=regularizer, padding='same', activation='relu')) 
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(64, 3, kernel_regularizer=regularizer, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(64, 5, kernel_regularizer=regularizer, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    
    model.add(Conv2D(128, 4, kernel_regularizer=regularizer, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model
def model_builder():
    input_shape = (28, 28, 1)
    regularizer = regularizers.l2(5e-4)
    
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape)) 
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(32, 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(SpatialDropout2D(0.25))
    
    model.add(Conv2D(64, 3, padding='same', activation='relu')) 
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Conv2D(64, 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    
    model.add(Conv2D(128, 4, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.25))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model
model = model_builder()
model.summary()
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
# Callbacks for single model
checkpoint_filepath = '~/checkpoint'

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
)
lr_annealer_callback = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
plot_losses_callback = PlotLossesKerasTF()
callbacks = [lr_annealer_callback, early_stopping_callback, plot_losses_callback]
batch_size = 64
steps_per_epoch = X_train.shape[0] // batch_size
train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
# Train a single model
history = model.fit(train_gen, batch_size=batch_size, epochs=55, steps_per_epoch=steps_per_epoch,
                    validation_data=(X_val, y_val), callbacks=callbacks)
# Plot losses and accuracies
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

skip_epochs = 0  # start plotting after this number of epochs
graph_epochs = range(skip_epochs, len(loss))

plt.subplot(2, 1, 1)
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(graph_epochs, loss[skip_epochs:], label='Training loss')
plt.plot(graph_epochs, val_loss[skip_epochs:], label='Validation loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.hlines(0.996, xmin=skip_epochs, xmax=len(loss))
plt.hlines(0.997, xmin=skip_epochs, xmax=len(loss))
plt.plot(graph_epochs, accuracy[skip_epochs:], label='Training accuracy')
plt.plot(graph_epochs, val_accuracy[skip_epochs:], label='Validation accuracy')
plt.legend()
plt.gcf().set_size_inches(15, 12)
# load best weights from save (this isn't always the best choice as it can load from when the model was underfitting)
model.load_weights(checkpoint_filepath)
final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"""
Final val loss: {final_loss:.5f}
Final val accuracy: {final_acc:.5f}
""")
predicted_classes = np.argmax(model.predict(X_test), axis=-1)
submission = pd.DataFrame({'ImageId': list(range(1, len(predicted_classes) + 1)),
                          'Label': predicted_classes})
submission.to_csv('mnist_cnn.csv', index=False)
files.download('mnist_cnn.csv')
# Ensemble of 15 nets
num_nets = 15
models = [model_builder() for _ in range(num_nets)]
batch_size = 64
epochs = 55
history = [0] * num_nets
for i in range(num_nets):
    checkpoint_filepath = f'~/checkpoint{i}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )
    lr_annealer_callback = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.93 ** x)
    callbacks = [lr_annealer_callback, model_checkpoint_callback]
    X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.1)
    train_gen = datagen.flow(X_train2, y_train2, batch_size=batch_size)
    steps_per_epoch = X_train2.shape[0] // batch_size
    history[i] = models[i].fit(train_gen, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=(X_val2, y_val2), callbacks=callbacks, verbose=0)
    train_acc, val_acc = max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])
    print(f'CNN {i + 1}: Epochs={epochs}, Best Train Acc={train_acc:.5f}, Best Val Acc={val_acc:.5f}')
# Check ensemble validation accuracy
results = np.zeros((X_val.shape[0], 10))
for i in range(num_nets):
    results += models[i].predict(X_val)
results = np.argmax(results, axis=1)
print('Val Acc:', np.mean(results == np.argmax(y_val, axis=1)))
# Find most common prediction for test set and save submission
results = np.zeros((X_test.shape[0], 10))
for i in range(num_nets):
    results += models[i].predict(X_test)
results = np.argmax(results, axis=1)
submission = pd.DataFrame({'ImageId': list(range(1, len(results) + 1)),
                          'Label': results})
submission.to_csv('mnist_cnn_ens.csv', index=False)
files.download('mnist_cnn_ens.csv')