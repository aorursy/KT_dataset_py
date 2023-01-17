import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, experimental, Dropout
from sklearn.model_selection import train_test_split
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.head()
# Spliting and shuffling our dataset.
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Forming np arrays of labels and features.

train_labels = np.array(train_df.pop('label'))
val_labels = np.array(val_df.pop('label'))

train_features=np.array(train_df)
val_features=np.array(val_df)
test_features=np.array(test_df)
height=28
width=28
num_classes=10

train_features=train_features.reshape(train_features.shape[0],height,width,1)
val_features=val_features.reshape(val_features.shape[0],height,width,1)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
val_labels = keras.utils.to_categorical(val_labels, num_classes)
print('Training features shape: ',train_features.shape)
print('Validation features shape: ',val_features.shape)

print('Training labels shape: ', train_labels.shape)
print('Validation labels shape: ', val_labels.shape)
plt.imshow(train_features[1].reshape(28, 28))
plt.title("labeled one-hot vector: {}".format(train_labels[1]))
model = Sequential([
  experimental.preprocessing.Rescaling(1./255, input_shape=(train_features.shape[1:])),
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.2),
  Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.summary()
epochs=20
batch_size=32

history = model.fit(
    train_features,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_features, val_labels))

test_features=test_features.reshape(test_features.shape[0],height,width,1)
predictions = np.argmax(model.predict(test_features),axis = 1)
predictions
submission_dataframe = pd.DataFrame({"ImageId" : range(1, test_features.shape[0]+1), "Label" : predictions})
submission_dataframe.to_csv("submission.csv", index = False)