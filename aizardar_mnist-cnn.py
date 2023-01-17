# Let's import some useful libraries

import pandas as pd  # 
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
import tensorflow as tf
import numpy as np

# Read csv files 

train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
train_df.head()
# Let's extract Label column from our training data

y = train_df['label']
train_df.drop(columns = ['label'], axis = 1, inplace = True)
# Let's see how many digit images we have in our training data

y.value_counts()
train_df = train_df.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)


for i in range(16):
  # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    plt.imshow(train_df[i][:,:,0])

plt.show()
y = to_categorical(y, num_classes = 10)

# Split our data into training and validation set

X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size = 0.1, random_state=2020, stratify = y)
# Define our CNN architecture

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (7,7), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

from tensorflow.keras.callbacks import EarlyStopping

early_stopper = EarlyStopping(monitor='val_accuracy', patience = 10, verbose=True, mode = "max" , restore_best_weights = True)


history = model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size = 100),
        steps_per_epoch=378,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks = [early_stopper]
        )
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# Let's now predict results
results = model.predict(test_df)

# We will select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("CNN_MNIST_predictions.csv",index=False)
