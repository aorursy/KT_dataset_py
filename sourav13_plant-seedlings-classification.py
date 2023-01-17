# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#visualization

import matplotlib.pyplot as plt

import seaborn as sns



#keras import

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, AveragePooling2D

from keras.layers import GlobalAveragePooling2D

from keras.utils.np_utils import to_categorical

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.preprocessing import image

from keras.optimizers import Adam, RMSprop, SGD

from keras.applications.resnet50 import ResNet50

from sklearn.utils import shuffle



from tqdm import tqdm



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
base_dir = '../input'

train_dir = os.path.join(base_dir, 'train')

test_dir = os.path.join(base_dir, 'test')
print(os.listdir(train_dir))
categories = ['Fat Hen', 'Black-grass', 'Cleavers', 'Small-flowered Cranesbill', 'Sugar beet',

              'Common Chickweed', 'Maize', 'Loose Silky-bent', 'Common wheat', 'Scentless Mayweed', 'Shepherds Purse', 'Charlock']

for category in categories:

    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))
train = []

for category_id, category in enumerate(categories):

    for imagefile in os.listdir(os.path.join(train_dir, category)):

        train.append(['train/{}/{}'.format(category, imagefile), category, category_id])



train = pd.DataFrame(train, columns=['filepath', 'category', 'category_id'])

print(train.shape)

print(train.head())
train = train.sample(frac=1).reset_index(drop=True) #Here frac=1 gives the entire dataframe jumbled up
print(train.head())
print(train.shape)
test = []

for imagefile in os.listdir(test_dir):

    test.append(['test/{}'.format(imagefile), imagefile])

test = pd.DataFrame(test, columns=['filepath', 'file'])
print(test.shape)
test.head(2)
def read_image(filepath, size):

    img = image.load_img(os.path.join(base_dir, filepath), target_size=size)

    img = image.img_to_array(img)

    return img

image_size = 100

#Taking X_train as numpy array and each element of the array is an array image

X_train = np.zeros((train.shape[0], image_size, image_size, 3), dtype='float32')

#Taking y_train to store the labels

y_train = np.zeros((train.shape[0], 1), dtype='float32')



for i, imagepath in tqdm(enumerate(train['filepath'])):

    #Reading the image

    img = read_image(imagepath, (image_size, image_size))

    #storing it 

    X_train[i] = img

    #storing the labels

    y_train[i] = train['category_id'][i]
print(X_train.shape)

print(y_train.shape)
#Here we are seeing 1st/0th image of X_train  

plt.imshow(X_train[0][:,:,1])

print(y_train[0])


plt.imshow(X_train[1][:,:,1])

print(y_train[1])
X_test = np.zeros((test.shape[0], image_size, image_size, 3), dtype='float32')
for i, imagefile in tqdm(enumerate(test['filepath'])):

    img = read_image(imagefile, (image_size, image_size))

    X_test[i] = img
print(X_test.shape)
plt.imshow(X_test[0][:,:,0])
y_train = to_categorical(y_train, num_classes=12)

print(y_train.shape)
print(y_train[0])
X_train = X_train / 255.0

X_test = X_test / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)
print(X_train.shape)

print(y_train.shape)

print('<===============================================================>')

print(X_val.shape)

print(y_val.shape)
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(image_size, image_size, 3)))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=128, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(BatchNormalization(axis=3))

model.add(Dropout(0.2))



model.add(Conv2D(filters=256, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(BatchNormalization(axis=3))

model.add(Dropout(0.2))



model.add(Conv2D(filters=512, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='Same', activation='relu'))



model.add(BatchNormalization(axis=3))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(12, activation='softmax'))



model.summary()
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
image_gen = ImageDataGenerator(rotation_range = 180, zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1,

                              horizontal_flip = True, vertical_flip = True)
image_gen.fit(X_train)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.5, verbose=1, min_lr=0.00001)
epochs = 70

batch_size = 100
history = model.fit_generator(image_gen.flow(X_train, y_train, batch_size=batch_size),

                                            epochs=epochs, verbose=2, callbacks=[learning_rate_reduction], validation_data=(X_val, y_val), 

                                            steps_per_epoch = X_train.shape[0] // batch_size)
accuracy = history.history['acc']

loss = history.history['loss']



val_accuracy = history.history['val_acc']

val_loss = history.history['val_loss']



epoch = range(len(accuracy))



plt.plot(epoch, accuracy, color='red', label='Training accuracy')

plt.plot(epoch, val_accuracy, label='Crossvalidation accuracy')

plt.title('Trainin accuracy vs Crossvalidation accuracy')

plt.legend()



plt.figure()

plt.plot(epoch, loss, color='red', label='Training loss')

plt.plot(epoch, val_loss, label='Crossvalidation loss')

plt.title('Trainin loss vs Crossvalidation loss')

plt.legend()



plt.show()
result = model.predict(X_test)



result = np.argmax(result, axis=1)
print(result)
test.head()
test['species'] = [categories[i] for i in result] 
test.head()
test[['file', 'species']].to_csv('submission.csv', index=False)