import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape
y_train = train['label']
train.drop(axis=1, labels=['label'], inplace=True)
train.head()
y_train.value_counts()    ## Images of 1's is maximum & Images of 5's is minimum
train /= 255.
test /= 255.
test = np.array(test).astype(np.float32)
X = np.array(train).astype(np.float32)
del train
X = X.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)
plt.imshow(X[0][:,:,0], cmap='gray')
print('label = '+str(y_train[0]))
yOHE = keras.utils.to_categorical(y_train, num_classes=10)
X_train, X_val, Y_train, Y_val = train_test_split(X, yOHE, test_size=0.1, random_state=2)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='Same', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='Same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='Same'))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='Same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(keras.layers.Conv2D(filters=96, kernel_size=(3,3), activation='relu', padding='Same'))
model.add(keras.layers.Conv2D(filters=96, kernel_size=(3,3), activation='relu', padding='Same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))
#optim = keras.optimizers.RMSprop(lr=0.001,rho=0.9, epsilon=1e-08, decay=0.0)
#optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001) ---> val_acc = 99.59
optim = keras.optimizers.Adadelta()
lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.00001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,                         # set input mean to 0 over the dataset
        samplewise_center=False,                          # set each sample mean to 0
        featurewise_std_normalization=False,              # divide inputs by std of the dataset
        samplewise_std_normalization=False,               # divide each input by its std
        zca_whitening=False,                              # apply ZCA whitening
        rotation_range=10,                                # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15,                                # Randomly zoom image 
        width_shift_range=0.1,                            # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,                           # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,                            # randomly flip images
        vertical_flip=False)                              # randomly flip images

datagen.fit(X_train)
filepath="BestWeights_Adadelta.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
callbacks_list.append(lr_reducer)
batch_size = 64
history = model.fit_generator(  datagen.flow(X_train, Y_train, batch_size=batch_size), 
                             epochs=30, 
                             validation_data=(X_val, Y_val),
                             steps_per_epoch=X_train.shape[0] // batch_size, 
                             callbacks=callbacks_list, 
                             verbose=1
                          )
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
model.load_weights('BestWeights_Adadelta.h5')
preds = model.predict(test, batch_size=batch_size, verbose = 1)
final_preds = np.argmax(preds, axis=1)
final_preds
plt.imshow(test[7,:,:,0], cmap='gray')
print(final_preds[7])
dictionary = {'Label':final_preds}
pred_df = pd.DataFrame(data=dictionary, index=list(range(1,len(final_preds)+1)))
pred_df.to_csv('submission.csv')
"""Accuracy on kaggle :- 99.66% ----> Top 10% """
