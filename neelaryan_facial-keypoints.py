import os
inputPath = os.path.join(os.getcwd(), '..', 'input')
print(os.listdir(inputPath))
inputPath = os.path.join(inputPath, 'facial-keypoints', 'data')
print(os.listdir(inputPath))
### Useful routines for preparing data
import numpy as np
import os
import pandas as pd
from os.path import join
import cv2

img_size = 100
def load_imgs_and_keypoints(dirname = inputPath):
    # Write your code for loading images and points here
    df = pd.read_csv(join(dirname, 'gt.csv'))
    dirname = join(dirname, 'images')
    total = len(df.index.values)
    cols = len(df.columns.values) - 1
    imgs = np.zeros((total, img_size, img_size, 3))
    points = np.zeros((total, cols))
    for i, row in df.iterrows() :
        img = cv2.imread(join(dirname, row['filename']))
        if i % 1000 == 0 : print(i + 1)
        pnts = np.asarray(row.to_list()[1:], dtype = np.float32)
        pnts[1::2] /= img.shape[0]
        pnts[0::2] /= img.shape[1]
        points[i] = pnts - 0.5
        img = cv2.resize(img, dsize = (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[i] = img / 255
    print(imgs.shape)
    return imgs, points
imgs, points = load_imgs_and_keypoints()
# Example of output
%matplotlib inline
from skimage.io import imshow
imshow(imgs[0])
points[0]
import matplotlib.pyplot as plt
# Circle may be useful for drawing points on face
# See matplotlib documentation for more info
from matplotlib.patches import Circle

def visualize_points(img, points):
    # Write here function which obtains image and normalized
    # coordinates and visualizes points on image
    fig, ax = plt.subplots()
    im = ax.imshow(img)
    x, y = points[0::2], points[1::2]
    x = (x + 0.5) * img_size
    y = (y + 0.5) * img_size
    plt.plot(x, y, 'o', color = 'red', markersize = 4.5)
    plt.show()
    
total = imgs.shape[0]
id = np.random.randint(0, total)
visualize_points(imgs[id], points[id])
from sklearn.model_selection import train_test_split
imgs_train, imgs_val, points_train, points_val = train_test_split(imgs, points, test_size=0.1)
print(imgs_train.shape)
print(imgs_val.shape)
print(points_train.shape)
print(points_val.shape)
def flip_img(img, img_points):
    # Write your code for flipping here
    fpts = img_points.copy()
    fpts[0::2] = -img_points[0::2]
    f_points = np.zeros((28))
    f_points[0:8:2] = np.flip(fpts[0:8:2],0)
    f_points[1:8:2] = np.flip(fpts[1:8:2],0)
    f_points[8:20:2] = np.flip(fpts[8:20:2],0)
    f_points[9:20:2] = np.flip(fpts[9:20:2],0)
    f_points[22:28:2] = np.flip(fpts[22:28:2],0)
    f_points[23:28:2] = np.flip(fpts[23:28:2],0)
    return np.flip(img,1), f_points

total = imgs.shape[0]
id = np.random.randint(0, total)
print('Image number : {}'.format(id))
visualize_points(*flip_img(imgs[id], points[id]))
visualize_points(imgs[id], points[id])
new_imgs, new_points = [], []
for img, img_points in zip(imgs_train, points_train) :
    f_img, f_points = flip_img(img, img_points)
    new_imgs.append(img)
    new_imgs.append(f_img)
    new_points.append(img_points)
    new_points.append(f_points)
    
aug_imgs_train = np.asarray(new_imgs, dtype = np.float32)
aug_points_train = np.asarray(new_points, dtype = np.float32)
print(aug_imgs_train.shape)
print(aug_points_train.shape)
total = imgs.shape[0]
id = np.random.randint(0, total)
print('Image number : {}'.format(id))
visualize_points(aug_imgs_train[2 * id], aug_points_train[2 * id])
visualize_points(aug_imgs_train[2 * id + 1], aug_points_train[2 * id + 1])
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import glorot_uniform
from keras import backend as K
from keras.regularizers import l2

K.clear_session()

model = Sequential()

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(28, activation='linear'))

model.summary()
# ModelCheckpoint can be used for saving model during training.
# Saved models are useful for finetuning your model 
# See keras documentation for more info
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop

# Choose optimizer, compile model and run training
BATCH_SIZE = 128
EPOCHS = 50
model_name = 'Facial_keypoints_model.h5'
checkpointer = ModelCheckpoint(filepath = model_name, monitor='val_accuracy', mode='max', save_best_only = True)
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
history = model.fit(
    aug_imgs_train, aug_points_train,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (imgs_val, points_val),
    callbacks = [checkpointer],
    shuffle = True,
    verbose = 1
)
# Plot history: MAE
plt.plot(history.history['loss'], label='MSE (training data)')
plt.plot(history.history['val_loss'], label='MSE (validation data)')
plt.title('The Graph (uncreative)')
plt.ylabel('MSE value')
plt.xlabel('No. of epochs')
plt.legend(loc="upper left")
plt.show()
model.save_weights(model_name)
model.load_weights(model_name)
val_points_pred = model.predict(imgs_val)
print(val_points_pred.shape)
total = imgs_val.shape[0]
ids = np.random.randint(0, total, size = 5)
for i in ids:
    visualize_points(imgs_val[i], val_points_pred[i])