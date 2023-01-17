import numpy as np 

import pandas as pd



data = pd.read_csv('../input/train.csv')

data.head(1)
from sklearn.model_selection import train_test_split

training_data, validation_data = train_test_split(data, test_size=.25, random_state=37)

print(training_data.shape, validation_data.shape)
y_train = training_data['label'].values

y_val = validation_data['label'].values

print(y_train.shape, y_val.shape)
x_train = np.reshape(training_data.drop('label', axis=1).values,(y_train.shape[0], 28,28))

x_val = np.reshape(validation_data.drop('label', axis=1).values,(y_val.shape[0], 28,28))

print(x_train.shape, x_val.shape)
from matplotlib import pyplot as plt



def preview(X,labels):

    n_wide = int(np.ceil(np.sqrt(len(labels))))

    fig, axes = plt.subplots(n_wide, n_wide, figsize=(12,12), subplot_kw={'xticks':[], 'yticks':[], 'frameon':False} )

    for i,x in enumerate(X):

        ax = axes[int(i/n_wide), int(i%n_wide)]

        if len(x.shape)==3: 

            x = x[:,:,0]

        ax.imshow(x)

        if type(labels[i]) is np.int64:

            ax.set_title(labels[i])

        else:

            ax.set_title(np.argmax(labels[i]))

        

preview(x_train[:30], y_train[:30])
from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

    

train_datagen = ImageDataGenerator(

        rotation_range=15, 

        height_shift_range=4, 

        width_shift_range=4, 

        shear_range=20, 

        data_format='channels_last',

    ).flow(np.expand_dims(x_train, axis=3), 

        y=to_categorical(y_train, dtype='int32'), 

        batch_size=128,

        shuffle=False,

        seed=37,

    )



val_datagen = ImageDataGenerator(

    data_format='channels_last',

    ).flow(np.expand_dims(x_val, axis=3), 

        y=to_categorical(y_val, dtype='int32'), 

        batch_size=128,

        shuffle=False,

        seed=37,

    )
x,y = train_datagen.next()

preview(x[:32],y[:32])
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint  



model = Sequential()

model.add(Conv2D(32, 2, input_shape=(28,28,1), activation='relu', padding='same'))

model.add(MaxPool2D(pool_size=2))

# model.add(Dropout(rate=.2, seed=37))

model.add(Conv2D(32, 2, padding='same', activation="relu"))

model.add(MaxPool2D(pool_size=2))

model.add(Dropout(rate=.2, seed=37))

model.add(Conv2D(64, 3, padding='same', activation="relu"))

model.add(MaxPool2D(pool_size=2))

model.add(Dropout(rate=.5, seed=37))

model.add(GlobalAveragePooling2D())

model.add(Dense(10, activation="softmax"))



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch_with_dropout.hdf5', 

                               verbose=1, save_best_only=True)

learning = model.fit_generator(train_datagen, 

        epochs=100, 

        steps_per_epoch=len(y_train)/train_datagen.batch_size,

        validation_data=val_datagen, 

        validation_steps=len(y_val)/val_datagen.batch_size,

        callbacks=[checkpointer],

    )
def show_learning_curves(history):

    plt.title("Accuracy")

    plt.plot(history['acc'])

    plt.plot(history['val_acc'])

    plt.xlabel('epoch')

    plt.legend(['Train', 'Test'], loc='lower right')

    plt.show()

    plt.title("Loss")

    plt.plot(history['loss'])

    plt.plot(history['val_loss'])

    plt.xlabel('epoch')

    plt.legend(['Train', 'Test'], loc='upper right')

    plt.show()

    plt.title("Log(Loss)")

    plt.plot(np.log(history['loss']))

    plt.plot(np.log(history['val_loss']))

    plt.xlabel('epoch')

    plt.legend(['Train', 'Test'], loc='upper right')

    plt.show()

show_learning_curves(learning.history)
model.load_weights('weights.best.from_scratch_with_dropout.hdf5')

predictions = [

    np.argmax(model.predict(np.expand_dims(np.expand_dims(x, axis=2), axis=0))) 

    for x in x_val ]



validation_accuracy = 100*np.sum(np.array(predictions)==y_val)/len(predictions)

print('Test accuracy: %.4f%%' % validation_accuracy)
test_data = pd.read_csv('../input/test.csv')

x_test = np.reshape(test_data.values,(len(test_data), 28,28, 1))

x_test.shape
test_predictions = [

    np.argmax(model.predict(np.expand_dims(x, axis=0)))

    for x in x_test ]

preview(x_test[:30], test_predictions[:30])
results_df = pd.DataFrame(test_predictions, columns=['Label'], index=np.arange(1,len(test_predictions)+1))

results_df.to_csv('results_submission.csv',index_label='ImageId')