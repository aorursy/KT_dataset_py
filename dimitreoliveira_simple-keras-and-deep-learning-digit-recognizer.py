import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
def plot_metrics_primary(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(20,7))

    ax1.plot(acc, label='Train Accuracy')
    ax1.plot(val_acc, label='Validation accuracy')
    ax1.legend(loc='best')
    ax1.set_title('Accuracy')

    ax2.plot(loss, label='Train loss')
    ax2.plot(val_loss, label='Validation loss')
    ax2.legend(loc='best')
    ax2.set_title('Loss')

    plt.xlabel('Epochs')
    
    
def plot_confusion_matrix(cnf_matrix, labels): 
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
    plt.figure(figsize=(20,7))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
SUBMISSION_NAME = 'submission.csv'

# Model parameters
BATCH_SIZE = 64
EPOCHS = 45
LEARNING_RATE = 0.001
HEIGHT = 28
WIDTH = 28
CANAL = 1
N_CLASSES = 10
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
labels = train['label']
train = train.drop(['label'], axis=1)
train.head(5)
train.shape
countplot(labels)
labels.value_counts()
# Reshape image in 3 dimensions (height, width, canal)
train = train.values.reshape(-1,HEIGHT,WIDTH,CANAL)
test = test.values.reshape(-1,HEIGHT,WIDTH,CANAL)
# Turn labels into np arrays
labels = labels.values
train.shape
for i in range(9):
    plt.subplot(330 + (i+1))
    plt.imshow(train[i][:,:,0], cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.title(labels[i]);
labels = pd.get_dummies(labels).values
train = train / 255.0
test = test / 255.0
x_train, x_val, y_train, y_val = train_test_split(train, labels, test_size=0.1, random_state=1)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15)  # randomly shift images vertically (fraction of total height)

datagen.fit(x_train)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same', input_shape=(HEIGHT, WIDTH, CANAL)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(N_CLASSES, activation = "softmax"))

optimizer = optimizers.adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])
print('Dataset size: %s' % train.shape[0])
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: (%s, %s, %s)' % (HEIGHT, WIDTH, CANAL))
x_train[0].shape
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
                              epochs=EPOCHS, validation_data=(x_val, y_val),
                              verbose=2, steps_per_epoch=x_train.shape[0] // BATCH_SIZE)
plot_metrics_primary(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss'])
cnf_matrix = confusion_matrix(np.argmax(y_val, axis=1), model.predict_classes(x_val))
plot_confusion_matrix(cnf_matrix, range(10))
predictions = model.predict_classes(test)
submission = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})
submission.to_csv(SUBMISSION_NAME, index=False)
submission.head(10)