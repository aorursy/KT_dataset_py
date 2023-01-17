import numpy as np

import pandas as pd

import os

import glob

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm_notebook as tqdm

%matplotlib inline

 

pd.set_option('display.max_rows', 10)
base_data_folder = "/kaggle/input/aptos2019-blindness-detection"

train_data_folder = os.path.join(base_data_folder, "train_images")



print(os.listdir(base_data_folder))
train_files_names = sorted(os.listdir(train_data_folder))

train_files_names[:6]
sorted(glob.glob(train_data_folder + '/*.png'))[:6]
train_images = []

for file in tqdm(sorted(glob.glob(train_data_folder + '/*.png'))):

    image_bgr = cv2.imread(file, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb = cv2.resize(image_rgb, dsize=None, fx = 0.15, fy=0.15, interpolation = cv2.INTER_AREA)

    train_images.append(image_rgb)
train_labels = pd.read_csv(base_data_folder+"/train.csv")

train_labels.sort_values(by='id_code', inplace = True)

train_labels
y_data = train_labels['diagnosis']

y_data[:5]
y_data.hist()

print(y_data.value_counts())
fig = plt.figure(figsize=(14,8))



for idx, image_rgb in enumerate(train_images[:10]):

    fig.add_subplot(2, 5, idx+1)

    plt.imshow(image_rgb)

    plt.title("Label:{0}".format(train_labels['diagnosis'][idx]))

    plt.xlabel(train_labels['id_code'][idx])

    plt.tight_layout()
def crop_image_from_gray(img,tol=7):

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray_img > tol

    

    img1=img[:,:,0][np.ix_(mask.any(axis=1),mask.any(axis=0))]

    img2=img[:,:,1][np.ix_(mask.any(axis=1),mask.any(axis=0))]

    img3=img[:,:,2][np.ix_(mask.any(axis=1),mask.any(axis=0))]

    img = np.stack([img1,img2,img3], axis=-1)

    

    return img
def circle_crop(img):

    img = crop_image_from_gray(img)

    height, width, depth = img.shape

    largest_side = np.max((height, width))

    

    img_reshaped = cv2.resize(img, dsize=(largest_side, largest_side))

                

    length = img_reshaped.shape[0]

    x = int(length/2)

    y = int(length/2)

    r = np.amin((x,y))    

    

    background = np.zeros_like(img_reshaped, dtype=np.uint8)

    circle_mask = cv2.circle(background, (x,y), int(r), (255, 255, 255), thickness=-1)

    

    image = cv2.bitwise_and(img_reshaped, circle_mask)

    

    return image
X_data = []

for image_rgb in tqdm(train_images):

    circle_img = circle_crop(image_rgb)

    image_resized = cv2.resize(circle_img, dsize=(224, 224))

    X_data.append(image_resized)
fig = plt.figure(figsize=(14,8))



for idx, image in enumerate(X_data[:10]):

    fig.add_subplot(2, 5, idx+1)

    plt.imshow(image)

    plt.title("Label:{0}".format(train_labels['diagnosis'][idx]))

    plt.xlabel(train_labels['id_code'][idx])

    plt.tight_layout()
X_data_prepocessed = []

for image_rgb in tqdm(X_data):

    blured = cv2.GaussianBlur(image_rgb, (9,9) ,15)

    image = cv2.addWeighted(image_rgb, 5, blured, -5, 128)

    X_data_prepocessed.append(image)
X_data_prepocessed = np.array(X_data_prepocessed)

X_data_prepocessed.shape
del train_images
fig = plt.figure(figsize=(14,8))



for idx, image in enumerate(X_data_prepocessed[:10]):

    fig.add_subplot(2, 5, idx+1)

    plt.imshow(image)

    plt.title("Label:{0}".format(train_labels['diagnosis'][idx]))

    plt.xlabel(train_labels['id_code'][idx])

    plt.tight_layout()
del X_data
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_data_prepocessed, y_data, test_size=0.2,

                                                      stratify = y_data, random_state = 123456)



print(X_train.shape, y_train.shape)

print(X_valid.shape, y_valid.shape)
from tensorflow.keras.utils import to_categorical



y_train_onehot = to_categorical(y_train, num_classes=5, dtype='bool')

y_valid_onehot = to_categorical(y_valid, num_classes=5, dtype='bool')



print(y_train_onehot.shape)

print(y_valid_onehot.shape)
plt.hist(y_train)

plt.hist(y_valid)

plt.title("Train and Validation set Distribution")

plt.legend(['Train', 'Validation'])

plt.show()
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'valid', input_shape = (224, 224, 3), name = 'Conv1-1'))

model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'Conv1-2'))

model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool1'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu',padding = 'valid', name = 'Conv2-1'))

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv2-2'))

model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool2'))
model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu',padding = 'valid', name = 'Conv3-1'))

model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv3-2'))

model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv3-3'))

model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool3'))
model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu',padding = 'valid', name = 'Conv4-1'))

model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv4-2'))

model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv4-3'))

model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool4'))
model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu',padding = 'valid', name = 'Conv5-1'))

model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv5-2'))

model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv5-3'))

model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu',padding = 'same', name = 'Conv5-4'))

model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool5'))
model.add(layers.Flatten())

model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, name='Dense1'))

model.add(layers.BatchNormalization())

model.add(layers.LeakyReLU(0.1))
model.add(layers.Dense(256, name='Dense2'))

model.add(layers.BatchNormalization())

model.add(layers.LeakyReLU(0.1))
model.add(layers.Dense(64, name='Dense3'))

model.add(layers.BatchNormalization())

model.add(layers.LeakyReLU(0.1))
model.add(layers.Dense(5, activation = 'softmax', name='Final')) # output layer
model.summary()
from tensorflow.keras import optimizers



model.compile(optimizer = optimizers.Adam(0.0001), loss = 'categorical_crossentropy', metrics=['acc'])
import time

from tensorflow.keras.callbacks import ModelCheckpoint



callback_list = [ModelCheckpoint(filepath='cnn_checkpoint.h5',

                                 monitor = 'val_loss',

                                 save_best_only = True)]
BATCH_SIZE = 256

num_epochs = 64



history = model.fit(X_train, y_train_onehot,

                    batch_size = BATCH_SIZE,

                    epochs = num_epochs,

                    validation_data = (X_valid, y_valid_onehot),

                    callbacks = callback_list)
epochs = np.arange(1, len(history.history['loss']) + 1)



plt.plot(epochs, history.history['loss'], label = 'Training')

plt.plot(epochs, history.history['val_loss'], label = 'Validation')

plt.title('Loss History Plot')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.plot(epochs, history.history['acc'], label = 'Training')

plt.plot(epochs, history.history['val_acc'], label = 'Validation')

plt.title('Accuracy History Plot')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
model.save('cnn_model.h5')
from tensorflow.keras.models import load_model



restored_model = load_model('cnn_model.h5')

restored_model.load_weights('cnn_checkpoint.h5')
restored_model.evaluate(X_valid, y_valid_onehot)
y_pred = np.argmax(restored_model.predict(X_valid), axis = 1)



print('Predict:', y_pred[:10])

print('Validation:', np.array(y_valid[:10]))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import cohen_kappa_score



cm = confusion_matrix(y_true = y_valid,

                      y_pred = y_pred)



kappa_score = cohen_kappa_score(y1 = y_valid,

                                y2 = y_pred,

                                weights='quadratic')



print("Confusion Matrix")

print(cm)

print()

print("Shape :", cm.shape)

print("Accurcy: {0:.2f}%".format(np.trace(cm) / np.sum(cm)*100))

print("Quadratic Weighted Kappa Score:", np.round(kappa_score, 4))
from sklearn.metrics import classification_report



print(classification_report(y_valid, y_pred, digits=4, target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']))
test_data_folder = os.path.join(base_data_folder, "test_images")

test_files_names = sorted(os.listdir(test_data_folder))

test_files_names[:5]
sorted(glob.glob(test_data_folder + '/*.png'))[:5]
test_images = []

for file in tqdm(sorted(glob.glob(test_data_folder + '/*.png'))):

    image_bgr = cv2.imread(file, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, dsize=None, fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)

    test_images.append(image_resized)
X_test = []

for image_rgb in tqdm(test_images):

    circle_img = circle_crop(image_rgb)

    image_resized = cv2.resize(circle_img, dsize=(224, 224))

    X_test.append(image_resized)
X_test_prepocessed = []

for image_rgb in tqdm(X_test):

    blured = cv2.GaussianBlur(image_rgb, (9,9) ,15)

    image = cv2.addWeighted(image_rgb, 5, blured, -5, 128)

    X_test_prepocessed.append(image)
fig = plt.figure(figsize=(14,8))



for idx, image_rgb in enumerate(test_images[:10]):

    fig.add_subplot(2, 5, idx+1)

    plt.imshow(image_rgb)

    plt.xlabel(test_files_names[idx])

    plt.tight_layout()
fig = plt.figure(figsize=(14,8))



for idx, image in enumerate(X_test[:10]):

    fig.add_subplot(2, 5, idx+1)

    plt.imshow(image)

    plt.xlabel(test_files_names[idx])

    plt.tight_layout()
fig = plt.figure(figsize=(14,8))



for idx, image in enumerate(X_test_prepocessed[:10]):

    fig.add_subplot(2, 5, idx+1)

    plt.imshow(image)

    plt.xlabel(test_files_names[idx])

    plt.tight_layout()
X_test = np.array(X_test_prepocessed)

X_test.shape
del test_images
test_image_labels = pd.DataFrame(columns = ['id_code'])



for i in test_files_names:

    splited = i.split('.')[0]

    temp = pd.DataFrame({'id_code':[splited]})

    test_image_labels = pd.concat([test_image_labels, temp], ignore_index=True)



test_image_labels
preds = np.argmax(model.predict(X_test), axis = 1)



print('Predicted:', preds[:10])
test_image_labels['diagnosis'] = pd.Series(preds)

test_image_labels
fig = plt.figure(figsize=(14,8))



for idx, image in enumerate(X_test[:20]):

    fig.add_subplot(4, 5, idx+1)

    plt.imshow(image)

    plt.title('diagnosed:{0}'.format(test_image_labels['diagnosis'][idx]))

    plt.xlabel(test_image_labels['id_code'][idx])

    plt.tight_layout()
plt.hist(test_image_labels['diagnosis'])

plt.title('Predicted class distribution')

plt.show()
test_image_labels.to_csv('submission.csv', index=False)
!ls -al