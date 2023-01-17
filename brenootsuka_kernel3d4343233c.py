import pandas as pd



from keras.utils import to_categorical



num_classes = 10



train_fname = '/kaggle/input/digit-recognizer/train.csv'

train_db = pd.read_csv(train_fname)



label = train_db['label'].to_numpy()

label = to_categorical(label, num_classes=num_classes)



data = train_db.drop('label', axis=1).to_numpy()

data = data.reshape((data .shape[0], 28, 28, 1))

data = data.astype('float32') / 255



x_train = data [:32000]

y_train = label[:32000]



x_valid = data [32000:]

y_valid = label[32000:]



del train_db
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rotation_range=10,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                   shear_range=0.1,

                                   zoom_range=0.1)

batch_size = 16



train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
from keras.models import Sequential

from keras.layers import Dense, Flatten, SeparableConv2D, Conv2D, MaxPooling2D

from keras.layers import SpatialDropout2D, Dropout, BatchNormalization

from keras.regularizers import l2



model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu'),

    SeparableConv2D(32, (5, 5), activation='relu', padding='same'),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    SpatialDropout2D(0.35),

    Conv2D(64, (3, 3), activation='relu'),

    Conv2D(64, (3, 3), activation='relu'),

    SeparableConv2D(64, (5, 5), activation='relu', padding='same'),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    SpatialDropout2D(0.35),

    Flatten(),

    Dense(256, activation='relu', kernel_regularizer=l2(1e-3)),

    Dense( 10, activation='softmax')

])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs = 30



performance = model.fit(train_generator,

                        steps_per_epoch=x_train.shape[0] // batch_size,

                        epochs=epochs,

                        validation_data=(x_valid, y_valid),

                        callbacks=[learning_rate_reduction])
import matplotlib.pyplot as plt



plt.subplots(figsize=(10, 12))



loss = performance.history["loss"]

val_loss = performance.history["val_loss"]



plt.subplot(211)

plt.title("Loss")

plt.plot(range(1, epochs + 1), loss, "bo-", label="Training Loss")

plt.plot(range(1, epochs + 1), val_loss, "ro-", label="Validation Loss")

plt.xticks(range(1, epochs + 1))

plt.grid(True)

plt.legend()



acc = performance.history["acc"]

val_acc = performance.history["val_acc"]



plt.subplot(212)

plt.title("Accuracy")

plt.plot(range(1, epochs + 1), acc, "bo-", label="Training Acc")

plt.plot(range(1, epochs + 1), val_acc, "ro-", label="Validation Acc")

plt.xticks(range(1, epochs + 1))

plt.grid(True)

plt.legend()



plt.show()
import seaborn as sns

import numpy as np



from sklearn.metrics import confusion_matrix



pred = model.predict(x_valid)

pred_classes = np.argmax(pred, axis=1)

pred_true = np.argmax(y_valid, axis=1)



confusion_mtx = confusion_matrix(pred_true, pred_classes)

sns.set(style="white", context="notebook", palette="deep")

sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap=plt.cm.Blues)



plt.show()
test_fname = '/kaggle/input/digit-recognizer/test.csv'



test_db = pd.read_csv(test_fname)



x_test = test_db.to_numpy()

x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_test = x_test.astype('float32') / 255



predictions = np.argmax(model.predict(x_test), axis=1)



imageIds = pd.Series(range(1,28001), name='ImageId')

results  = pd.Series(predictions   , name='Label'  )



submission = pd.concat([imageIds, results], axis=1)

submission.to_csv("submission.csv", index=False, header=True)
