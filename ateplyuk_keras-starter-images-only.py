import cv2

import pandas as pd

import numpy as np

import os

import json

from tqdm import tqdm, tqdm_notebook

from keras.models import Sequential

from keras.layers import Dense, Flatten, Activation

from keras.layers import Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.optimizers import SGD, RMSprop, Adam

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train/train.csv')

img_size = 128
def resize_to_square(im):

    old_size = im.shape[:2]

    ratio = float(img_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])    

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = img_size - new_size[1]

    delta_h = img_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)

    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

    return new_im
def load_image(path, pet_id):

    image = cv2.imread(f'{path}{pet_id}-1.jpg')

    new_image = resize_to_square(image)

    return new_image
im = load_image("../input/train_images/", "86e1089a3")

plt.imshow(im)
pet_ids = train_df['PetID'].values
%%time

X = []

Y = []

for pet_id in tqdm_notebook(pet_ids):

    try:

        im = load_image("../input/train_images/", pet_id)

        X.append(im)        

        ads = train_df[train_df['PetID'] == pet_id]['AdoptionSpeed'].values[0]

        Y.append(ads)

    except:

        pass  

X = np.asarray(X)
from sklearn.model_selection import train_test_split

X_tr, X_tst, y_tr, y_tst = train_test_split(X, Y, test_size=0.33, random_state=42)
X_tr = X_tr.astype('float32')

X_tst = X_tst.astype('float32')

X_tr /= 255

X_tst /= 255
batch_size = 32

nb_classes = 5

nb_epoch = 50
Y_tr = np_utils.to_categorical(y_tr, nb_classes)

Y_tst = np_utils.to_categorical(y_tst, nb_classes)
# Model



model = Sequential()

model.add(Conv2D(img_size, (3, 3), padding='same',

                        input_shape=(img_size, img_size, 3), activation='relu'))

model.add(Conv2D(img_size, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(img_size*2, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(img_size*2, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))



# opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
%%time

# Train model

history = model.fit(X_tr, Y_tr,

              batch_size=batch_size,

              epochs=nb_epoch,

              validation_split=0.1,

              shuffle=True,

              verbose=2)
# Evaluation

scores = model.evaluate(X_tst, Y_tst, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
test_df = pd.read_csv('../input/test/test.csv')

pet_ids_tst = test_df['PetID'].values

sam_sub_df = pd.read_csv('../input/test/sample_submission.csv')

print(sam_sub_df.shape)

sam_sub_df.head()
sub_dict = {'PetID': [], 'filename': []}



for name in os.listdir('../input/test_images'):

    pet_id = name.split('-')[0]

    sub_dict['PetID'].append(pet_id)

    sub_dict['filename'].append(name)

    

sub_df = pd.DataFrame(sub_dict)

sub_df.head()
%%time

test_img = [] # resized images (test)

pet_id_test_img = [] # ids of resized images (test)



for pet_id in tqdm_notebook(pet_ids_tst):

    try:

        im = load_image("../input/test_images/", pet_id)

        test_img.append(im)  

        

        pet_id_test_img.append(pet_id)

    except:

        pass 

test_img = np.asarray(test_img)

test_img = test_img.astype('float32')

test_img /= 255
%%time

# Prediction

test_predictions = model.predict(test_img)
sub_df = pd.DataFrame(test_predictions)

sub_df['PetID'] = pd.Series(pet_id_test_img, index=sub_df.index)

cols = sub_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

sub_df=sub_df[cols]

print(sub_df.shape)

sub_df.head()
sub_df['AdoptionSpeed'] = sub_df.iloc[:,1:6].values.argmax(axis=1)

sub_df.drop(columns=[0,1,2,3,4], inplace=True)

print(sub_df.shape)

sub_df.head()
res_df = sam_sub_df.join(sub_df.set_index('PetID'), on='PetID', rsuffix='_pred')

res_df.drop(columns=['AdoptionSpeed'],inplace=True)

res_df = res_df.rename(columns={'AdoptionSpeed_pred': 'AdoptionSpeed'})

res_df.fillna(0, inplace=True)

res_df['AdoptionSpeed'] = res_df['AdoptionSpeed'].astype(int)

print(res_df.shape)

res_df.head()
res_df.to_csv('submission.csv',index=False)