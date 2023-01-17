import numpy as np 

import pandas as pd 

import os



from collections import defaultdict

from glob import glob

from random import choice, sample

from keras.preprocessing import image

import cv2

from tqdm import tqdm_notebook

import numpy as np

import pandas as pd

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract

from keras.models import Model

from keras.optimizers import Adam
!pip install git+https://github.com/rcmalli/keras-vggface.git
from keras_vggface.utils import preprocess_input

from keras_vggface.vggface import VGGFace
train_file_path = "../input/train_relationships.csv"

train_folders_path = "../input/train/"
# train_dirs = os.listdir('../input/train/')

# len(train_dirs), train_dirs
val_famillies_list = ["F07", "F08", "F09"]

# val_famillies_list = ["F09"]
%%time

all_images = glob(train_folders_path + "*/*/*.jpg")
def get_train_val(family_name):

    # Get val_person_image_map

    val_famillies = family_name

    train_images = [x for x in all_images if val_famillies not in x]

    val_images = [x for x in all_images if val_famillies in x]



    train_person_to_images_map = defaultdict(list)



    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]



    for x in train_images:

        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)



    val_person_to_images_map = defaultdict(list)



    for x in val_images:

        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

        

    # Get the train and val dataset

    relationships = pd.read_csv(train_file_path)

    relationships = list(zip(relationships.p1.values, relationships.p2.values))

    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]



    train = [x for x in relationships if val_famillies not in x[0]]

    val = [x for x in relationships if val_famillies in x[0]]

    

    return train, val, train_person_to_images_map, val_person_to_images_map
def read_img(path):

    img = image.load_img(path, target_size=(197, 197))

    img = np.array(img).astype(np.float)

    return preprocess_input(img, version=2)



def gen(list_tuples, person_to_images_map, batch_size=16):

    ppl = list(person_to_images_map.keys())

    while True:

        batch_tuples = sample(list_tuples, batch_size // 2)

        labels = [1] * len(batch_tuples)

        while len(batch_tuples) < batch_size:

            p1 = choice(ppl)

            p2 = choice(ppl)



            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:

                batch_tuples.append((p1, p2))

                labels.append(0)



        for x in batch_tuples:

            if not len(person_to_images_map[x[0]]):

                print(x[0])



        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]

        X1 = np.array([read_img(x) for x in X1])



        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]

        X2 = np.array([read_img(x) for x in X2])



        yield [X1, X2], labels



import tensorflow as tf

from keras import backend as K

def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed





def baseline_model():

    input_1 = Input(shape=(197, 197, 3))

    input_2 = Input(shape=(197, 197, 3))



    base_model = VGGFace(model='resnet50', include_top=False)



    for x in base_model.layers[:-3]:

        x.trainable = True

    for x in base_model.layers[-3:]:

        x.trainable=False



    x1 = base_model(input_1)

    x2 = base_model(input_2)



#     x1_ = Reshape(target_shape=(7*7, 2048))(x1)

#     x2_ = Reshape(target_shape=(7*7, 2048))(x2)

#     #

#     x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])

#     x_dot = Flatten()(x_dot)



    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])

    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])



    x3 = Subtract()([x1, x2])

    x3 = Multiply()([x3, x3])



    x1_ = Multiply()([x1, x1])

    x2_ = Multiply()([x2, x2])

    x4 = Subtract()([x1_, x2_])

    x = Concatenate(axis=-1)([x4, x3])



    x = Dense(100, activation="relu")(x)

    x = Dropout(0.01)(x)

    out = Dense(1, activation="sigmoid")(x)



    model = Model([input_1, input_2], out)

    

    # loss="binary_crossentropy"

    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], 

                  metrics=['acc'], 

                  optimizer=Adam(0.00003))



    model.summary()



    return model
model = baseline_model()
n_val_famillies_list = len(val_famillies_list)
n_val_famillies_list
for i in tqdm_notebook(range(n_val_famillies_list)):

    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[i])

    file_path = f"vgg_face_{i}.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.5, patience=10, verbose=1)

    es = EarlyStopping(monitor="val_acc", min_delta = 0.001, patience=30, verbose=1)

    callbacks_list = [checkpoint, reduce_on_plateau, es]



    history = model.fit_generator(gen(train, train_person_to_images_map, batch_size=32), 

                                  use_multiprocessing=True,

                                  validation_data=gen(val, val_person_to_images_map, batch_size=32), 

                                  epochs=300, verbose=1,

                                  workers=4, callbacks=callbacks_list, 

                                  steps_per_epoch=400, validation_steps=200)
test_path = "../input/test/"



submission = pd.read_csv('../input/sample_submission.csv')



def chunker(seq, size=32):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
preds_for_sub = np.zeros(submission.shape[0])

for i in tqdm_notebook(range(n_val_famillies_list)):

    file_path = f"vgg_face_{i}.h5"

    model.load_weights(file_path)

    # Get the predictions

    predictions = []



    for batch in tqdm_notebook(chunker(submission.img_pair.values)):

        X1 = [x.split("-")[0] for x in batch]

        X1 = np.array([read_img(test_path + x) for x in X1])



        X2 = [x.split("-")[1] for x in batch]

        X2 = np.array([read_img(test_path + x) for x in X2])



        pred = model.predict([X1, X2]).ravel().tolist()

        predictions += pred

    preds_for_sub += np.array(predictions) / n_val_famillies_list
submission['is_related'] = preds_for_sub

submission.to_csv("vgg_face.csv", index=False)
submission