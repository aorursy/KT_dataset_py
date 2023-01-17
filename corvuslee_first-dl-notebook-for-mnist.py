import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import tensorflow as tf
tf.__version__
img_rows, img_cols = 28, 28
num_classes = 10  #0,1,2,3,4,5,6,7,8,9
train_file = "../input/train.csv"
test_file = "../input/test.csv"

train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')
print(train_data.shape, test_data.shape)
def prep_data(raw):
    out_y = 0
    out_x = 0
    if raw.shape[1] == 785:  # training dataset
        y = raw[:, 0]  # first column is label
        out_y = tf.keras.utils.to_categorical(y, num_classes)  # Converts a class vector (integers) to binary class matrix.
        x = raw[:,1:]
    elif raw.shape[1] == 784:  # testing dataset
        x = raw
    num_images = raw.shape[0]
    x /= 255  # scaling from 0-255 to 0-1
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    return out_x, out_y
X, y = prep_data(train_data)
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y)
print(X_train.shape, X_cv.shape, y_train.shape, y_cv.shape)
X_test, y_test = prep_data(test_data)
print(X_test.shape)
size=6
fig, axes = plt.subplots(1, size, figsize=(img_rows,img_cols))
for i,j in enumerate(np.random.randint(0,X.shape[0],size)):
    plt.subplot(1, size, i+1)
    plt.title("Label: %s" % (y[j]))
    plt.imshow(X[j].reshape(img_rows,img_cols),cmap="gray")
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32,  # number of filters
                 3,  # dimension of filters
                 activation="relu",
                 padding="same",  # pad with 0 to the edges to make output shapes = input shapes
                 input_shape=(img_rows, img_cols, 1)
                 ))
model.add(MaxPool2D(pool_size=2))  # pool and halve the size. i.e. 28x28 -> 14x14
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation="softmax"))
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(lr=3e-3),
              loss = tf.keras.losses.categorical_crossentropy,
              metrics = ["accuracy"])
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=3,
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
history = model.fit(X_train,
                    y_train,
                    batch_size = 512,
                    epochs = 50,
                    callbacks = [learning_rate_reduction],
                    validation_data=(X_cv, y_cv)
                    )
import gc
gc.collect()
plt.title("Learning rate curves")
plt.plot(history.history["lr"])
plt.ylabel("learning rate")
plt.xlabel("epochs")
plt.show()
plt.title("Loss curves")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(['train','cv'])
plt.show()
preds_cv = model.predict(X_cv)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_cv,axis=1), np.argmax(preds_cv,axis=1))
cm
df = pd.DataFrame({"expected": np.argmax(y_cv,axis=1), "predicted": np.argmax(preds_cv,axis=1)})
df_incorrect = df[df["expected"] != df["predicted"]]
df_incorrect.head()
for expected in range(10):
    index = df_incorrect[df_incorrect["expected"] == expected].index
    size=len(index)

    fig, axes = plt.subplots(1, size, figsize=(img_rows,img_cols))
    for i,j in enumerate(index):
        plt.subplot(1, size, i+1)
        plt.title("Predicted: %s" % (np.argmax(preds_cv[j])))
        plt.imshow(X_cv[j].reshape(img_rows,img_cols),cmap="gray")
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=10,
                                   shear_range=0.4)
cv_datagen = ImageDataGenerator(rotation_range=10,
                                shear_range=0.4)
from tensorflow.python.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10)
history = model.fit_generator(train_datagen.flow(X_train, y_train),
                              epochs=30,
                              callbacks = [learning_rate_reduction, early_stopping],
                              validation_data=cv_datagen.flow(X_cv, y_cv),
                              workers=2,
                              use_multiprocessing=True
                             )
import gc
gc.collect()
plt.title("Learning rate curves")
plt.plot(history.history["lr"])
plt.ylabel("learning rate")
plt.xlabel("epochs")
plt.show()
plt.title("Loss curves")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(['train','cv'])
plt.show()
preds_cv = model.predict(X_cv)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_cv,axis=1), np.argmax(preds_cv,axis=1))
cm
predictions = model.predict(X_test)
predictions.shape
!head ../input/sample_submission.csv
submission = pd.DataFrame({"ImageId": range(1,predictions.shape[0]+1), "Label": np.argmax(predictions, axis=1)})
submission.to_csv('submission.csv', index=False)
!head submission.csv