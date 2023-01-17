import pandas as pd
import numpy as np

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
# Read train.csv file
train = pd.read_csv("../input/train.csv")

train.head()
# Drop the labels
X_train = train.drop(labels="label", axis=1)
y_train = train["label"].values

# Reshape the images from pixels to image
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Normalize
X_train = X_train/255
y_train[0]
y_train = to_categorical(y_train, num_classes=10)
y_train[0]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Add
from keras.layers import BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# Input
inputs = Input(shape=(28, 28, 1))

bn0 = BatchNormalization(scale=True)(inputs)

# Initial Stage
conv1 = Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', kernel_initializer='uniform')(bn0)
conv1 = Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', kernel_initializer='uniform')(conv1)
bn1 = BatchNormalization(scale=True)(conv1)
max_pool1 = MaxPooling2D(pool_size=(2,2))(bn1)

# First
conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(max_pool1)
conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv2)
conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv2)
bn2 = BatchNormalization(scale=True)(conv2)

# First Residual
res_conv1 = Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer='uniform')(max_pool1)
res_bn1 = BatchNormalization(scale=True)(res_conv1)

# First Add
add1 = Add()([res_bn1, bn2])

# First Acvtivation & MaxPooling
act1 = Activation('relu')(add1)
max_pool2 = MaxPooling2D(pool_size=(2,2))(act1)

# Second
conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(max_pool2)
conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv3)
conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv3)
bn3 = BatchNormalization(scale=True)(conv3)

# Second Residual
res_conv2 = Conv2D(64, kernel_size=(3,3), padding='same', kernel_initializer='uniform')(max_pool2)
res_bn2 = BatchNormalization(scale=True)(res_conv2)

# Second Add
add2 = Add()([res_bn2, bn3])

# Second Acvtivation & MaxPooling
act2 = Activation('relu')(add2)
max_pool3 = MaxPooling2D(pool_size=(2,2))(act2)

# Flattern the data
flatten = Flatten()(max_pool3)

# Fully Connected Layer
dense1 = Dense(128, activation='relu')(flatten)
do = Dropout(0.25)(dense1)

dense2 = Dense(10, activation='softmax')(do)

model = Model(inputs=[inputs], outputs=[dense2])

# Parameters for training
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
datagen = ImageDataGenerator(rotation_range=25,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

datagen.fit(X_train)
# Checkpoint to save the best model
checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)

# Reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=1)

batch_size = 32
epochs = 40

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val,y_val),
                              verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[checkpointer, reduce_lr])
!ls
from keras.models import load_model
model = load_model("weights.17-0.005.hdf5")
test = pd.read_csv("../input/test.csv")

X_test = test/255

X_test = X_test.values.reshape(-1, 28, 28, 1)
results = model.predict(X_test)

# Convert the results into labels
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("resnet_mnist.csv",index=False)
