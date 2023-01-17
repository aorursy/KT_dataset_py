%pylab

import glob

from scipy.ndimage import imread

from scipy.ndimage.interpolation import zoom

from scipy.misc import imresize

import pandas as pd

import seaborn as sns

sns.set_style('white')

%matplotlib inline
import zipfile

z = zipfile.ZipFile('../input/BDRW_train/BDRW_train_1.zip', "r")

for name in z.namelist():

    z.extract(name)

z = zipfile.ZipFile('../input/BDRW_train/BDRW_train_2.zip', "r")

for name in z.namelist():

    z.extract(name)
labels = pd.read_excel('BDRW_train_2/labels.xls')

labels.columns = ['digit', 'label']
labels.head()
labels.info()
labels.label.hist()
dummies = pd.get_dummies(labels, columns = ['label'], )

one_hot_labels =array(dummies.drop('digit', axis = 1))
image_list = []

# bring in order according to labels.xls

for filename in sorted(glob.glob('BDRW_train_1/*.jpg'), key=lambda a: int(a.split("_")[-1].split('.')[0]))[1:]: #because digit_0 not in labels

    im=imread(filename)

    image_list.append(array(im))

for filename in sorted(glob.glob('BDRW_train_2/*.jpg'), key=lambda a: int(a.split("_")[-1].split('.')[0]))[:]:

    im=imread(filename)

    image_list.append(array(im))
shapes = array([x.shape for x in image_list]).T

scatter(shapes[0], shapes[1])
mean(shapes[:2], 1)
def resize_and_fill(im, X=35, Y = 28):

    i = argmin(im.shape[:2])

    i = im.shape[0]/float(im.shape[1]) >= 1.25

    if i == 0:

        test_im = imresize(im, (int(im.shape[0]*float(Y)/im.shape[1]) , Y, 3) )

    if i == 1:

        test_im = imresize(im, (X, int(im.shape[1]*float(X)/im.shape[0]), 3) )



    shape_test = array(test_im.shape[:2])

    test_im2 = ones((X, Y ,3))

    if i == 0:

        for k in range(3):

            test_im2[:, :, k] = concatenate((test_im[:,:,k], ones((X-shape_test[0], Y))*mean(test_im[-1,:,k])), 0)



    if i == 1:

        for k in range(3):

            test_im2[:, :, k] = concatenate((test_im[:,:,k], ones((X, Y-shape_test[1]))*mean(test_im[:,-1,k])), 1)

    return test_im2.astype(uint8)
adjusted_im = array([resize_and_fill(x) for x in image_list])
adjusted_im2 = append(adjusted_im, array([x[:,:,[1,0,2]] for x in adjusted_im]), 0)

one_hot_labels2 = append(one_hot_labels, one_hot_labels, 0)
fig, ax = subplots(5, 5, figsize = (14, 10))

for x in range(5):

    for y in range(5):

        rn = random_integers(0, len(adjusted_im))

        ax[x,y].imshow(adjusted_im[rn])

        ax[x,y].set_title(labels.label.iloc[rn])

show()
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import keras.backend as K



earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
K.clear_session()
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



model = Sequential()

model.add(Dropout(0.3, input_shape=((35, 28, 3))))

model.add(Conv2D(32, (3, 3), input_shape=(35, 28, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))
model.add(Flatten())  #3D feature maps to 1D feature vectors

model.add(Dense(64, kernel_regularizer= 'l2'))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10, kernel_regularizer= 'l2'))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'],)

#             callbacks = [earlyStopping])
model.summary()
X_train, X_test, y_train, y_test = train_test_split(adjusted_im2, one_hot_labels2, test_size = .1)
%%time



hist = model.fit(X_train, y_train, epochs = 150, batch_size = 64, verbose = False)
figure()

xlabel('epochs')

plot(hist.history['loss'])

ylabel('loss')

legend(['loss'])

twinx(gca())

plot(hist.history['acc'], 'r')

ylabel('accuracy')

legend(['acc'])

show()
print('loss, accuracy = ', model.evaluate(X_test, y_test))