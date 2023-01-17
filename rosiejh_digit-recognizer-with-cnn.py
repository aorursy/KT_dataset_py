import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.head()
test.head()
train.describe()
train.isnull().any().describe()
test.isnull().any().describe()
X_train = train.iloc[:, 1:].values.astype('float32') / 255  # Normalization
y_train = train.iloc[:, :1].values.astype('int32')  # 1st column is 'label' for images
X_test = test.values.astype('float32') / 255  # Normalization
X_train
y_train
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # Reshaping
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)
print(X_test.shape)
# Show first 10 images with their labels
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(y_train[i, 0])
# Convert list to One-hot encoded matrix
y_train = to_categorical(y_train)

# For example, '3' would be [0,0,0,1,0,0,0,0,0,0]
y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
model = Sequential([
    Conv2D(32, (5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(32, (5, 5), padding='Same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), padding='Same', activation='relu'),
    Conv2D(64, (3, 3), padding='Same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(lr=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_lr=0.00001)
datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.3)  
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                              epochs=50,
                              validation_data=datagen.flow(X_val, y_val, batch_size=128),
                              verbose=1,
                              steps_per_epoch=X_train.shape[0] // 64,
                              callbacks=[reduce_lr])
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
epochs = range(1, len(acc_values)+1)

plt.plot(epochs, acc_values, linestyle='-')
plt.plot(epochs, val_acc_values, linestyle=':')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()
predictions = model.predict_classes(X_test, verbose=0)

submissions = pd.DataFrame({"ImageId" : list(range(1, len(predictions)+1)), "Label" : predictions})
submissions.to_csv("digit_recognizer.csv", index=False, header=True)