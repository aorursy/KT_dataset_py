import numpy as np

import os

from math import ceil

import glob

import re

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import random

from skimage.util.shape import view_as_blocks

from skimage import io, transform

from sklearn.model_selection import train_test_split, KFold

from keras.utils.vis_utils import plot_model

from keras import layers, models, optimizers

from keras.initializers import he_normal, lecun_normal

from keras import backend as K

import keras

import warnings



from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



from tqdm import tqdm_notebook as tqdm





warnings.filterwarnings('ignore')
TESTING=False



SQUARE_SIZE = 40#must be less than 400/8==50

train_size = 3000

test_size = 500

BATCH_SIZE= 128

SEED=2019

Epoch=100

k_folds=5

PATIENCE=5

if TESTING:

    train_size=2000

    test_size=100

    BATCH_SIZE=64

    SEED=SEED

    Epoch=2

    k_folds=3

    PATIENCE=2
random.seed(SEED)

from numpy.random import seed

seed(SEED)

from tensorflow import set_random_seed

set_random_seed(SEED)
DATA_PATH='../input/dataset'

TRAIN_IMAGE_PATH=os.path.join(DATA_PATH, 'train')

TEST_IMAGE_PATH=os.path.join(DATA_PATH, 'test')
def get_image_filenames(image_path, image_type):

    if(os.path.exists(image_path)):

        return glob.glob(os.path.join(image_path, '*.'+image_type))

    return
train = get_image_filenames(TRAIN_IMAGE_PATH, "jpeg")#train 이미지 이름 리스트

test = get_image_filenames(TEST_IMAGE_PATH, "jpeg")#test 이미지 이름 리스트



random.shuffle(train)

random.shuffle(test)



train = train[:train_size]

test = test[:test_size]



piece_symbols = 'prbnkqPRBNKQ'
def fen_from_filename(filename):

  base = os.path.basename(filename)

  return os.path.splitext(base)[0]
print(fen_from_filename(train[0]))

print(fen_from_filename(train[1]))

print(fen_from_filename(test[2]))
f, axarr = plt.subplots(1,3, figsize=(120, 120))



for i in range(0,3):

    axarr[i].set_title(fen_from_filename(train[i]), fontsize=70, pad=30)

    axarr[i].imshow(mpimg.imread(train[i]))

    axarr[i].axis('off')
def onehot_from_fen(fen):

    eye = np.eye(13)

    output = np.empty((0, 13))

    fen = re.sub('[-]', '', fen)



    for char in fen:

        if(char in '12345678'):

            output = np.append(output, np.tile(eye[12], (int(char), 1)), axis=0)

        else:

            idx = piece_symbols.index(char)

            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)



    return output



def fen_from_onehot(one_hot):

    output = ''

    for j in range(8):

        for i in range(8):

            if(one_hot[j][i] == 12):

                output += ' '

            else:

                output += piece_symbols[one_hot[j][i]]

        if(j != 7):

            output += '-'



    for i in range(8, 0, -1):

        output = output.replace(' ' * i, str(i))



    return output
def process_image(img):

    downsample_size = SQUARE_SIZE*8

    square_size = SQUARE_SIZE

    img_read = io.imread(img)

    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')

    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))

    tiles = tiles.squeeze(axis=2)

    return tiles.reshape(64, square_size, square_size, 3)
def train_gen(features, batch_size):

    i=0

    while True:

        batch_x=[]

        batch_y=[]

        for b in range(batch_size):

            if i==len(features):

                i=0

                random.shuffle(features)

            img=str(features[i])

            y = onehot_from_fen(fen_from_filename(img))

            x = process_image(img)

            for x_part in x:

                batch_x.append(x_part)

            for y_part in y:

                batch_y.append(y_part)

            i+=1

        yield (np.array(batch_x), np.array(batch_y))



def pred_gen(features, batch_size):

    for i, img in enumerate(features):

        yield process_image(img)
#def train_gen(features, batch_size):

#    x_batch=[]

#    y_batch=[]

#    cnt=0

#    while True:

#        for i, img in enumerate(features):

#            y_batch += onehot_from_fen(fen_from_filename(img))

#            x_batch += process_image(img)

#        yield (np.array(x_batch), np.array(y_batch))

#

#def pred_gen(features, batch_size):

#    for i, img in enumerate(features):

#        yield process_image(img)
def get_callbacks(model_name, patient):

    ES = EarlyStopping(

        monitor='val_loss', 

        patience=patient, 

        mode='min', 

        verbose=1)

    RR = ReduceLROnPlateau(

        monitor = 'val_loss', 

        factor = 0.5, 

        patience = patient / 2, 

        min_lr=0.000001, 

        verbose=1, 

        mode='min')

    MC = ModelCheckpoint(

        filepath=model_name, 

        monitor='val_loss', 

        verbose=1, 

        save_best_only=True, 

        mode='min')



    return [ES, RR, MC]
def weighted_categorical_crossentropy(weights):

    """

    A weighted version of keras.objectives.categorical_crossentropy

    

    Variables:

        weights: numpy array of shape (C,) where C is the number of classes

    

    Usage:

        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

        loss = weighted_categorical_crossentropy(weights)

        model.compile(loss=loss,optimizer='adam')

    """

    

    weights = K.variable(weights)

        

    def loss(y_true, y_pred):

        # scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc

        loss = y_true * K.log(y_pred) * weights

        loss = -K.sum(loss, -1)

        return loss

    

    return loss
def get_model(image_size):#(model_name, image_size)

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(image_size, image_size, 3)))

    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(0.2))

    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(13, activation='softmax', kernel_initializer='lecun_normal'))

    

#    model.summary()

    

    weights=np.array([1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(64-10)])

    model.compile(loss=weighted_categorical_crossentropy(weights), optimizer='nadam', metrics=['acc'])#weight the inverse of expected pieces

    

    return model
#model=get_model(DenseNet121, SQUARE_SIZE)

#model=get_model(SQUARE_SIZE)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
kf = KFold(n_splits=k_folds, random_state=SEED)
j = 1

model_names=[]

for (train_fold, valid_fold) in kf.split(train):

    print("=========================================")

    print("====== K Fold Validation step => %d/%d =======" % (j,k_folds))

    print("=========================================")

    

    model_name = '../'+ str(j) + '.hdf5'

    model_names.append(model_name)

    model=get_model(SQUARE_SIZE)

    

    history=model.fit_generator(train_gen([train[i] for i in tqdm(train_fold)], batch_size=BATCH_SIZE), steps_per_epoch=ceil(train_size*(1-1/k_folds)/BATCH_SIZE), epochs=Epoch, validation_data=train_gen([train[i] for i in tqdm(valid_fold)], batch_size=BATCH_SIZE), validation_steps=ceil(train_size/k_folds/BATCH_SIZE), verbose=1, shuffle=False, callbacks=get_callbacks(model_name, PATIENCE))

    j+=1#single batch is actually 64*batch_size, since there are 64 pieces on the board
for name in tqdm(model_names):

    res = (

      (keras.models.load_model(name, custom_objects={'loss':weighted_categorical_crossentropy(np.array([1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(64-10)]))})).predict_generator(pred_gen(test, 64), steps=test_size)

      .argmax(axis=1)

      .reshape(-1, 8, 8)

    )

    pred_fens = np.array([fen_from_onehot(one_hot) for one_hot in res])

    test_fens = np.array([fen_from_filename(fn) for fn in test])

    

    final_accuracy = (pred_fens == test_fens).astype(float).mean()

    

    print("Model Name: ", name, "Final Accuracy: {:1.5f}%".format(final_accuracy))
def load_all_models(names):

    models=[]

    for model_name in names:

        models.append(keras.models.load_model(model_name, custom_objects={'loss':weighted_categorical_crossentropy(np.array([1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(64-10)]))}))

    return models
def get_stacked_model(models):

    input_layer=keras.layers.Input(shape=(SQUARE_SIZE, SQUARE_SIZE, 3,))

    xs=[model(input_layer) for model in models]

    out=keras.layers.Add()(xs)

    

    model=keras.models.Model(inputs=[input_layer], outputs=out)

    return model
model=get_stacked_model(load_all_models(model_names))
res_stacked = (

  model.predict_generator(pred_gen(test, 64), steps=test_size)

  .argmax(axis=1)

  .reshape(-1, 8, 8)

)
pred_fens = np.array([fen_from_onehot(one_hot) for one_hot in res_stacked])

test_fens = np.array([fen_from_filename(fn) for fn in test])



final_accuracy = (pred_fens == test_fens).astype(float).mean()



print("Final Accuracy: {:1.5f}%".format(final_accuracy))
def display_with_predicted_fen(image):

    pred = model.predict(process_image(image)).argmax(axis=1).reshape(-1, 8, 8)

    fen = fen_from_onehot(pred[0])

    imgplot = plt.imshow(mpimg.imread(image))

    plt.axis('off')

    plt.title(fen)

    plt.show()
display_with_predicted_fen(test[0])

display_with_predicted_fen(test[1])

display_with_predicted_fen(test[2])