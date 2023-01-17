import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bson,io

from random import randint

from skimage.data import imread
data = bson.decode_file_iter(open('../input/train.bson', 'rb'))

'''

    data generator for keras fit_generator function

'''



def convert2onehot(category_id):

    # you should have a dictionary for mapping each id to an index number

    # for demonstration purposes, I will only include a dummy file

    return True,category_id



def data_generator(batch_size=128, start_image=0):

    count_product = 0

    images = []

    y_label = []

    while True:

        count = 0

        for c, d in enumerate(data):

            category_id = d['category_id']

            if count_product < start_image:

                count_product += 1

                continue

            success, one_hot = convert2onehot(category_id) # be sure to create your own y_label output

            if not success:

                print("id conversion failed")

                continue

            for e, pic in enumerate(d['imgs']):

                picture = imread(io.BytesIO(pic['picture']))

                images.append(picture)

                y_label.append(one_hot)

                count += 1

            if count >= batch_size:

                count = 0

                y_label = np.asarray(y_label)

                images = np.asarray(images)

                '''

                    since shuffle in fit function will not work here, 

                    a batch shuffle mechnism is added 

                '''

                for i,image in enumerate(images[:int(batch_size/2)]):

                    j = randint(0,batch_size-1)

                    y_temp = y_label[i]

                    img_temp = image

                    images[i] = images[j]

                    y_label[i] = y_label[j]

                    images[j] = img_temp

                    y_label[j] = y_temp

                yield images, y_label

                # just to be sure past batch are removed from the memory

                del images

                del y_label

                images = []

                y_label = []

                

                

'''

    Example: 

    models.fit_generator(data_generator(BATCH_SIZE),steps_per_epoch=70000,nb_epoch=NB_EPOCH,callbacks=[validate_callback],workers=1)

    Note: steps per epoch is the total iteration in one epoch

'''
train_gen = data_generator()

next(train_gen)  # warm-up



%time bx, by = next(train_gen)


import pickle

from keras.callbacks import Callback



class ValidateCallbacks(Callback):

    # validation data is preprocessed and saved as pickle file

    def __init__(self):

        self.test_data = pickle.load(open("data/validation_data.p","rb"))



    def on_epoch_end(self, epoch, logs=None):

        images,y_label = self.test_data

        y_pred = self.model.predict(images)

        batch_size = len(y_pred)

        count = 0

        for index, pred in enumerate(y_pred):

            y_id, max_prob = get_max(pred)

            if y_id == y_label[index]:

                count += 1

        print("  Validation Score {}".format((count * 1.0) / (batch_size * 1.0)))