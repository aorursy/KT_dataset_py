import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import pickle



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
# constants

IMG_SIZE = 28

N_CHANNELS = 1 # because gray scale images
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

pred_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train_df = train_df.append(test_df)
train_df.head()
print (f'Training set: {train_df.shape}')

print (f'To be Predicted: {pred_df.shape}')
X_train = train_df.drop(['label'], axis = 1)

Y_train = train_df['label']

X_pred = pred_df.drop(['id'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.15)
X_train, X_test, X_pred = X_train.apply(lambda x: x/255), X_test.apply(lambda x: x/255), X_pred.apply(lambda x: x/255)
Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)
X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

X_pred = X_pred.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
print (f'Training images: {X_train.shape}')

print (f'Testing images: {X_test.shape}')
Y_train = Y_train.to_numpy()
fig, ax = plt.subplots(nrows=3, ncols=4)

count=0

for row in ax:

    for col in row:

        col.set_title(np.argmax(Y_train[count, :]))

        col.imshow(X_train[count, :, :, 0])

        count += 1

plt.show()
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





# This will just calculate parameters required to augment the given data. This won't perform any augmentations

datagen.fit(X_train)
models = [0] * 4 # Model array to store different types of CNN architectures



for model_type in range(len(models)):

    models[model_type] = Sequential()

    models[model_type].add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(28, 28, 1)))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    if model_type > 0:

        models[model_type].add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

        models[model_type].add(MaxPool2D(pool_size=(2, 2)))

        models[model_type].add(BatchNormalization(momentum=0.15))

    elif model_type > 1:

        models[model_type].add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))

        models[model_type].add(MaxPool2D(pool_size=(2, 2)))

        models[model_type].add(BatchNormalization(momentum=0.15))

    elif model_type > 2:

        models[model_type].add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))

        models[model_type].add(MaxPool2D(pool_size=(2, 2)))

        models[model_type].add(BatchNormalization(momentum=0.15))

    models[model_type].add(Flatten())

    models[model_type].add(Dense(128, activation = "relu"))

    models[model_type].add(Dense(10, activation = "softmax"))
# Compile all the models

for model_type in range(len(models)):

    models[model_type].compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
batch_size=32

epochs = 25
histories = [0] * len(models)

for model_type in range(len(models)):

    # Fit the models

    print (f'### Training model # {model_type}')

    histories[model_type] = models[model_type].fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                                  epochs = epochs, validation_data = (X_test,Y_test),

                                  steps_per_epoch=X_train.shape[0] // batch_size, 

                                  callbacks=[learning_rate_reduction])
for model_type in range(len(models)):

    models[model_type].save(f'model_{model_type}.h5')
for i in range(len(models)):

    with open(f'his{i}.pkl', 'wb') as output:

        pickle.dump(histories[i], output, pickle.HIGHEST_PROTOCOL)
%matplotlib inline

def PlotLoss(histories, epochs, is_training):

    plt.style.use("ggplot")

    plt.figure()

    for i in range(len(histories)):

        if is_training:

            plt.plot(np.arange(0, epochs), histories[i].history["loss"], label=f"train_loss_{i}")

            plt.title("Training Loss")

        else:

            plt.plot(np.arange(0, epochs), histories[i].history["val_loss"], label=f"val_loss_{i}")

            plt.title("Validation Loss")

    plt.xlabel("Epoch #")

    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    plt.show()



def PlotAcc(histories, epochs, is_training):

    plt.style.use("ggplot")

    plt.figure()

    for i in range(len(histories)):

        if is_training:

            plt.plot(np.arange(0, epochs), histories[i].history["accuracy"], label=f"train_acc_{i}")

            plt.title("Training Accuracy")

        else:

            plt.plot(np.arange(0, epochs), histories[i].history["val_accuracy"], label=f"val_acc_{i}")

            plt.title("Validation Accuracy")

    plt.xlabel("Epoch #")

    plt.ylabel("Accuracy")

    plt.legend(loc="lower right")

    plt.show()
PlotLoss(histories, epochs, True)

PlotLoss(histories, epochs, False)

PlotAcc(histories, epochs, True)

PlotAcc(histories, epochs, False)
models = [0] * 4 # Model array to store different types of CNN architectures



for model_type in range(len(models)):

    models[model_type] = Sequential()

    

    models[model_type].add(Conv2D(filters=2**(model_type + 3), kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(28, 28, 1)))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    

    models[model_type].add(Conv2D(filters=2**(model_type + 4), kernel_size=(3, 3), padding='SAME', activation='relu'))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    

    models[model_type].add(Conv2D(filters=2**(model_type + 5), kernel_size=(3, 3), padding='SAME', activation='relu'))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    

    models[model_type].add(Flatten())

    models[model_type].add(Dense(128, activation = "relu"))

    

    models[model_type].add(Dense(10, activation = "softmax"))
# Compile all the models

for model_type in range(len(models)):

    models[model_type].compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
batch_size=32

epochs = 25
histories = [0] * len(models)

for model_type in range(len(models)):

    # Fit the models

    print (f'### Training model # {model_type}')

    histories[model_type] = models[model_type].fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                                  epochs = epochs, validation_data = (X_test,Y_test),

                                  steps_per_epoch=X_train.shape[0] // batch_size, 

                                  callbacks=[learning_rate_reduction])
for model_type in range(len(models)):

    models[model_type].save(f'model_{model_type}.h5')
for i in range(len(models)):

    with open(f'his{i}.pkl', 'wb') as output:

        pickle.dump(histories[i], output, pickle.HIGHEST_PROTOCOL)
PlotLoss(histories, epochs, True)

PlotLoss(histories, epochs, False)

PlotAcc(histories, epochs, True)

PlotAcc(histories, epochs, False)
models = [0] * 6 # Model array to store different types of CNN architectures



for model_type in range(len(models)):

    models[model_type] = Sequential()

    

    models[model_type].add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(28, 28, 1)))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    

    models[model_type].add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    

    models[model_type].add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))

    models[model_type].add(MaxPool2D(pool_size=(2, 2)))

    models[model_type].add(BatchNormalization(momentum=0.15))

    

    models[model_type].add(Flatten())

    models[model_type].add(Dense(2**(model_type + 5), activation = "relu"))

    

    models[model_type].add(Dense(10, activation = "softmax"))
# Compile all the models

for model_type in range(len(models)):

    models[model_type].compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
batch_size=32

epochs = 25
histories = [0] * len(models)

for model_type in range(len(models)):

    # Fit the models

    print (f'### Training model # {model_type}')

    histories[model_type] = models[model_type].fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                                  epochs = epochs, validation_data = (X_test,Y_test),

                                  steps_per_epoch=X_train.shape[0] // batch_size, 

                                  callbacks=[learning_rate_reduction])
for model_type in range(len(models)):

    models[model_type].save(f'model_{model_type}.h5')

    with open(f'his{model_type}.pkl', 'wb') as output:

        pickle.dump(histories[model_type], output, pickle.HIGHEST_PROTOCOL)
PlotLoss(histories, epochs, True)

PlotLoss(histories, epochs, False)

PlotAcc(histories, epochs, True)

PlotAcc(histories, epochs, False)
preds = models[3].predict(X_pred)
pred_df['label'] = np.argmax(preds, axis=1)

preds = pred_df[['id', 'label']]

preds.to_csv('sub.csv', index=False)