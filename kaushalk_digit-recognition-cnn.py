import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
X_train, x_val, y_train, y_val = train_test_split(

    train.iloc[:,1:], train.iloc[:,0], test_size=0.1)

X_train.shape,x_val.shape
print(X_train.shape)

X_train.head()
print(test.shape)

test.head()
y_train.value_counts()


plt.figure(figsize=(15,7))

g = sns.countplot(y_train, palette="icefire")

plt.title("Number of digit classes")

img = X_train.iloc[1]

img = np.asarray(img)

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[3,0])

plt.axis("off")

plt.show()
img = X_train.iloc[3]

img = np.asarray(img)

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[3,0])

plt.axis("off")

plt.show()
X_train = X_train / 255.0

x_val = x_val / 255.0

print("x_train shape: ",X_train.shape)

print("test shape: ",x_val.shape)
# Reshape

X_train = X_train.values.reshape(-1,28,28,1)

x_val = x_val.values.reshape(-1,28,28,1)

print("x_train shape: ",X_train.shape)

print("test shape: ",x_val.shape)
# Label Encoding 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)

y_val = to_categorical(y_val, num_classes = 10)
plt.imshow(X_train[1][:,:,0],cmap='gray')

plt.show()
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# Define the optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



epochs = 9# for better result increase the epochs

batch_size = 90
# data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.5,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.5, # Randomly zoom image 5%

        width_shift_range=0.5,  # randomly shift images horizontally 5%

        height_shift_range=0.5,  # randomly shift images vertically 5%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=X_train.shape[0] // batch_size,callbacks=[learning_rate_reduction])
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
y_hat = model.predict(x_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(cm)
test = test.values.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(test, batch_size=90)
y_pred = np.argmax(y_hat,axis=1)
submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submission['Label'] = y_pred

submission.head(10)
submission.to_csv("submission.csv", index=False, header=True)