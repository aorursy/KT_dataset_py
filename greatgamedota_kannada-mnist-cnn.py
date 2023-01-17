import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam
class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

batch_size = 32

num_epochs = 30



# Import data

data_train_file = "../input/Kannada-MNIST/train.csv"

data_test_file = "../input/Kannada-MNIST/test.csv"

df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)



# Prep data

X_train = df_train.drop(labels=["label"], axis=1) / 255

Y_train = df_train["label"]

X_test = df_test.drop(labels=["id"], axis=1) / 255



Y_train = to_categorical(Y_train)



num_classes = Y_train.shape[1]



X_train = np.array(X_train).reshape(X_train.shape[0], 28, 28, 1)

X_test = np.array(X_test).reshape(X_test.shape[0], 28, 28, 1)

print(X_train.shape)
# Show image of training data

plt.figure(figsize=(10, 10))

rand_indexes = np.random.randint(0, X_train.shape[0], 8) # select random indices

for index,im_index in enumerate(rand_indexes):

    plt.subplot(4, 4, index + 1)

    plt.imshow(X_train[im_index].reshape(28, 28), cmap='gray', interpolation='none')

    plt.title('Class ' + str(int(class_names[np.argmax(Y_train[im_index])]) - 1))

plt.tight_layout()
# Build model

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(.25))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(.5))

model.add(Dense(num_classes, activation='softmax'))

# Compile Model

opt = Adam(lr=0.0001, beta_1=0.95, beta_2=0.99)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Train Model

model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
# Save model predictions as submission

submission = pd.DataFrame(range(0, df_test.shape[0]), columns=['id'])

submission['label'] = model.predict_classes(X_test, batch_size=batch_size)

submission.head(5000)
filename = 'submission.csv'

submission.to_csv(filename, index=False)