import numpy as np

import pandas as pd

from keras.applications import ResNet50

from keras.layers import GlobalAveragePooling2D, Dropout, Dense

from keras.models import Model

import keras

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from math import ceil, floor

import cv2

from tqdm import tqdm_notebook as tqdm

from PIL import Image

import matplotlib.pyplot as plt

from keras_efficientnet import EfficientNetB0
BASE_DIR = '/kaggle/input/bengaliai-cv19/'
def get_resized_dataset(data, size=(64, 64)):

    resized_data = []

    for arr in tqdm(data):

        resized_img = cv2.resize(arr.reshape(137,236), size, interpolation = cv2.INTER_AREA)

        resized_data.append(resized_img.reshape(-1))

    return np.array(resized_data)
def get_test_batch():

    for i in range(4):

        # load train.csv

#         test_df = pd.read_csv(BASE_DIR + 'test.csv')

        test_df = pd.read_parquet(BASE_DIR + f'test_image_data_{i}.parquet')



#         test_df = pd.concat([pq_df_0, pq_df_1, pq_df_2, pq_df_3], ignore_index=True)

        test_image_ids = test_df['image_id'].values        

        test_df.drop(columns=['image_id'], inplace=True)

        final_test_data = get_resized_dataset(test_df.values, size=(128, 128))



        del test_df

        

        yield final_test_data, test_image_ids
class DataTestGenerator(keras.utils.Sequence):



    def __init__(self, data, batch_size=1, img_size=(128, 128, 1), *args, **kwargs):



        self.data = data

        self.list_IDs = np.arange(data.shape[0])

        self.batch_size = batch_size

        self.img_size = img_size

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.indices) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indices]

        

        return self.__data_generation(list_IDs_temp)



        

    def on_epoch_end(self):

        self.indices = np.arange(len(self.list_IDs))



    def __data_generation(self, list_IDs_temp):

        X = np.empty((len(list_IDs_temp), *self.img_size))

        for i, ID in enumerate(list_IDs_temp):

                X[i,] = self.data[ID].reshape(*self.img_size)



        

        return X
# base_model = ResNet50(weights= None, include_top=False, input_shape= (64, 64, 1))

# base_model.trainable = False



# x = GlobalAveragePooling2D()(base_model.output)

# x = Dropout(0.25)(x)

# x = Dense(1000, activation='relu')(x)

# x = Dropout(0.25)(x)

# x = Dense(256, activation='relu')(x)



# vowel_diacritic_fc1 = Dense(11, activation='softmax', name='vowel_diacritic')(x)

# grapheme_root_fc1 = Dense(168, activation='softmax', name='grapheme_root')(x)

# consonant_diacritic_fc1 = Dense(7, activation='softmax', name='consonant_diacritic')(x)



# model = Model(inputs = base_model.input, outputs = [vowel_diacritic_fc1, \

#                                                     grapheme_root_fc1, \

#                                                     consonant_diacritic_fc1])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

base_model =  EfficientNetB0(weights =None, include_top = False, \

                                 pooling = 'avg', input_shape = (128, 128, 1))

x = base_model.output

# x = GlobalAveragePooling2D()(base_model.output)

x = Dropout(0.25)(x)

x = Dense(1000, activation='relu')(x)

x = Dropout(0.25)(x)

x = Dense(256, activation='relu')(x)



vowel_diacritic_fc1 = Dense(11, activation='softmax', name='vowel_diacritic')(x)

grapheme_root_fc1 = Dense(168, activation='softmax', name='grapheme_root')(x)

consonant_diacritic_fc1 = Dense(7, activation='softmax', name='consonant_diacritic')(x)



model = Model(inputs = base_model.input, outputs = [vowel_diacritic_fc1, \

                                                    grapheme_root_fc1, \

                                                    consonant_diacritic_fc1])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.load_weights('/kaggle/input/bairesnet50/model_effnet_b0.h5')
preds = []

image_ids = []

for sample, test_image_ids in get_test_batch():    

    test_gen = DataTestGenerator(sample/255, batch_size=64)

    print(test_gen.__len__())

    # model.load_weights(filepath)

    vowel_diacritic, grapheme_root, consonant_diacritic = model.predict_generator(test_gen, \

                                                                                  verbose=1)

    for i in range(len(test_image_ids)):

        image_ids.append(f"{test_image_ids[i]}_consonant_diacritic")

        image_ids.append(f"{test_image_ids[i]}_grapheme_root")

        image_ids.append(f"{test_image_ids[i]}_vowel_diacritic")



        preds.append(np.argmax(consonant_diacritic[i]))

        preds.append(np.argmax(grapheme_root[i]))

        preds.append(np.argmax(vowel_diacritic[i]))
subm_df = pd.DataFrame()

subm_df['row_id'] = image_ids

subm_df['target'] = preds

subm_df.head()
subm_df.to_csv('submission.csv', index=False)
subm_df.shape