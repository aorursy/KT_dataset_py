import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout

from keras.optimizers import RMSprop, Adam, SGD

from keras import regularizers

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

train.head(5)
print('Train set: ',train.shape, "\t Test Set", test.shape)
y = train['label']

# droping label cloumn in training set

train.drop('label', axis=1, inplace=True)
g = sns.countplot(y)
plt.figure(figsize=(12,5))

for i in range(40):

    plt.subplot(4,10,i+1)

    img = train.iloc[i,:].values.reshape(28,28)

    plt.imshow(img)

    plt.axis('off')

plt.tight_layout()

plt.show()
from keras.utils.np_utils import to_categorical 

y = to_categorical(y, num_classes = 10)

y[0]
train = train.values.reshape(train.shape[0], 28, 28, 1)

test = test.values.reshape(test.shape[0], 28, 28, 1)

print('Reshaped Train set: ',train.shape, " & Reshaped Test Set", test.shape)
train = train.astype("float32")/255.0

test = test.astype("float32")/255.0
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.25, random_state=0)



print("Number of samples in Training set :", X_train.shape[0])

print("Number of samples in Validation set :", X_val.shape[0])
train_datagen = ImageDataGenerator(rotation_range=10,

                                   zoom_range=0.1,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1

                                  )



training_set = train_datagen.flow(X_train, y_train,

                                  batch_size=64

                                 )



val_datagen = ImageDataGenerator()

val_set = val_datagen.flow(X_val, y_val,

                           batch_size=64

                          )
model = tf.keras.models.Sequential()



model.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy']

             )

model.summary()
# If the model is not improving on validation, we need to reduce the learning rate, If val loss is not improved in 4 epoch then lr will be reduced 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 

                              factor=0.2, 

                              patience=4, 

                              verbose=1, 

                              min_delta=0.0001)
steps_per_epoch = training_set.n // training_set.batch_size

validation_steps = val_set.n // val_set.batch_size



hist = model.fit(x=training_set,

                 validation_data=val_set,

                 epochs=35,

                 callbacks=[reduce_lr],

                 steps_per_epoch=steps_per_epoch,

                 validation_steps=validation_steps

                )
plt.figure(figsize=(14,5))

plt.subplot(1,2,2)

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['train', 'test'], loc='upper left')



plt.subplot(1,2,1)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
_, acc_val = model.evaluate(val_set)

_, acc_tr = model.evaluate(val_set)

print("\nFinal Accuracy on training set : {:.2f}% & accuracy on validation is set: {:.2f}%".format(acc_tr*100, acc_val*100))
from keras.utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True)
val_pred = model.predict(val_set)

val_pred = np.argmax(val_pred, axis=1)

y_val = np.argmax(y_val, axis=1)



from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix")

cm = confusion_matrix(y_val, val_pred)

print(cm)

print("Classification Report")

print(classification_report(y_val, val_pred))



#g = sns.heatmap(cm, cmap='Blues')

plt.figure(figsize=(8,8))

plt.imshow(cm, interpolation='nearest')

plt.colorbar()

target_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

tick_mark = np.arange(len(target_names))

_ = plt.xticks(tick_mark, target_names)

_ = plt.yticks(tick_mark, target_names)
pred = model.predict(test)

res = np.argmax(pred, axis=1)

submission = pd.DataFrame({"ImageId":[i+1 for i in range(len(test))],

                           "Label": res})

submission.head(10)
submission.to_csv("submission.csv", index=False)