import tensorflow as tf



from keras.models import Sequential

from keras.applications import VGG16

from keras.activations import relu, tanh

from keras.optimizers import Adagrad

from keras.losses import binary_crossentropy

from keras.layers import Dropout, Dense, Conv1D, MaxPool1D, Activation, Flatten, Input, Conv2D, MaxPooling2D, MaxPool2D,BatchNormalization



from skimage import io

from keras.models import Model

from skimage import io

from keras.optimizers import Adam, Adadelta, Adam, RMSprop



import cv2, io



from keras.utils import to_categorical 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



import matplotlib.pyplot as plt

import seaborn as sns



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train.head()
trainImageList = os.listdir("../input/train/train/")

trainImageList[:5]
trainImageList[0]
import skimage, sklearn



image =  skimage.io.imread("../input/train/train/0014d7a11e90b62848904c1418fc8cf2.jpg", as_gray=False)

image.shape
from skimage.filters import roberts, sobel, scharr, prewitt



# edge_roberts = roberts(image)

# edge_sobel = sobel(image)



# plt.imshow(edge_sobel);
plt.imshow(image)

plt.title('Cactus picture')

plt.show()
sns.countplot(train['has_cactus']);
cnt = 0

for i in train[16+30:25+30].values:

    cnt += 1

    image = skimage.io.imread(f"../input/train/train/{i[0]}", as_gray=False)

    plt.subplot(250+cnt)

    plt.axis("off")

    plt.imshow(image)

    plt.title(['no_cactus','has_cactus'][i[1]])

plt.show()
train.shape, image.shape
Y = to_categorical(num_classes=2, y=train['has_cactus'])

Y.shape, Y[:10]
num_classes = 2

epochs = 5

batch_size = 128

image.shape, train.columns
inputs = Input(shape=(32, 32, 3))



# vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

conv1 = Conv2D(256, (4, 4), activation='elu')(inputs)

batch1 = BatchNormalization()(conv1)

pool1 = MaxPooling2D(padding='same', pool_size=(2,2))(batch1)



conv2 = Conv2D(128, (4, 4), activation='relu')(pool1)

batch2 = BatchNormalization()(conv2)

pool2 = MaxPooling2D(padding='valid', pool_size=(3,3))(conv2)



conv3 = Conv2D(64, (3, 3), activation='elu')(pool2)

batch3 = BatchNormalization()(conv3)

pool3 = MaxPooling2D(padding='same', pool_size=(3,3))(batch3)



flatten = Flatten()(pool3)



dropout = Dropout(rate = 0.7)(flatten)



output = Dense(2, activation='sigmoid')(dropout)



model = Model(inputs=inputs, outputs=output)
# vgg_conv.summary()
# model = Sequential()

# # Add the vgg convolutional base model

# model.add(vgg_conv)

 

# # Add new layers

# model.add(Flatten())

# model.add(Dense(1024, activation='relu'))

# model.add(Dropout(1 - .70))

# model.add(Dense(2, activation='sigmoid'))
model.summary()
train['d'] = train['id'].apply(lambda x : "../input/train/train/"+x)

train['d'].head()
# train['data'] = train['d'].apply(lambda y: cv2.imread(y))

# train['data'].head()

Y.shape
from skimage import io



all_images = []

for image_path in train['d'].values:

#     print(image_path)

    img = io.imread(image_path, as_gray=False)

    img = img.reshape([32, 32,3])

    all_images.append(img/255)

X = np.array(all_images)

X.shape, type(X)
X[0].shape

optimizer = Adagrad(lr=0.001, decay=1e-3)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#Splits

# X_val = x_train[16000:]

# Y_val = Y[16000:]



x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=123)

x_train.shape, y_train.shape
from keras.callbacks import EarlyStopping

monitor = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=8)

model.fit(x=x_train, y=y_train, epochs=32, batch_size=512, shuffle=True, 

          validation_split=0.2, callbacks=[monitor])
score = model.evaluate(x_val, y_val)

score
# a = np.array([[1.88852847e-01, 9.69758689e-01],

#        [9.99574542e-01, 1.15156174e-04],

#        [3.27082276e-02, 9.89496231e-01]])

# np.where(a > 0.75)
preds = np.array(model.predict(x_val, batch_size=32))

preds = pd.DataFrame(preds)



res = preds[0].apply(lambda x : 0 if x >0.75 else 1)
print(classification_report(np.argmax(y_val, axis=1), res))
c = confusion_matrix(np.argmax(y_val, axis=1), res)

c
sns.heatmap(c, annot=True);
testImageList = os.listdir("../input/test/test/")

# testImageList
img = cv2.imread("../input/test/test/f0720f7eac8fd0b72dd78cc7f63f4467.jpg")

img.shape
test_images = []

for image_path in os.listdir("../input/test/test"):

#     print(image_path)

    img = skimage.io.imread('../input/test/test/'+image_path, as_gray=False)

    img = img.reshape([32, 32,3])

    test_images.append(img/255)

x_test = np.array(test_images)

x_test.shape
predTest1 = model.predict(x_test, batch_size=128)



predTest1 = pd.DataFrame(predTest1)



predTest1 = preds[0].apply(lambda x : 0 if x >0.75 else 1)

predTest1
submission = pd.read_csv("../input/sample_submission.csv")

submission.head()
# predTest = np.argmax(predTest1, axis=1)
submission['has_cactus'] = pd.Series(predTest1)
submission.to_csv("submission5.csv", index=False)

submission.head()
sns.countplot(submission['has_cactus']);
submission[46:55].values
cnt = 0



for i in submission[46:55].values:

    cnt += 1

    image = skimage.io.imread(f"../input/test/test/{i[0]}", as_gray=False)

    plt.subplot(250+cnt)

    plt.axis("off")

    plt.imshow(image)

    plt.title(['no_cactus','has_cactus'][int(i[1])])

plt.show()