# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalAveragePooling2D, Dense

from tqdm import tqdm

from tensorflow.keras.models import Model



from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import cv2

import gc


# for i in range(1):

#     df_test1 = pd.read_parquet("test_image_data_0.parquet")

#     df_test2 = pd.read_parquet("test_image_data_1.parquet")



#     df_test3 = pd.read_parquet("test_image_data_2.parquet")



#     df_test4  = pd.read_parquet("test_image_data_3.parquet")

# df_test = pd.concat([df_test1, df_test2, df_test3, df_test4 ], axis = 0)
df_class_map = pd.read_csv("/kaggle/input/bengaliai-cv19/" + "class_map.csv")

df_test_csv = pd.read_csv("/kaggle/input/bengaliai-cv19/"+ "test.csv")

df_train_csv = pd.read_csv("/kaggle/input/bengaliai-cv19/" + "train.csv")

df_submission = pd.read_csv("/kaggle/input/bengaliai-cv19/" + "sample_submission.csv")
Temp = np.empty((0, 64, 64,1))

Temp_Label = np.empty((0,1))





def Train_Data_Prepration(Temp = Temp, Temp_Label = Temp_Label):



    for i in ["train_image_data_0.parquet", "train_image_data_1.parquet", "train_image_data_2.parquet", "train_image_data_3.parquet"]:

        

        df_train1 = pd.read_parquet("/kaggle/input/bengaliai-cv19/" + i)



        df_train1 = shuffle(df_train1)



        Label = df_train1[['image_id']]

                                    

        df_train = df_train1.iloc[:, 1:].values



        del df_train1

        gc.collect()



        Array = np.empty((len(df_train), 64, 64,1))



        for i in tqdm(range(len(df_train))):

            a = df_train[i, :].reshape(137,236)#

            b = a.copy()

            a[a > 230] = 255

            b = b[:, ~np.all(a[1:] == a[:-1], axis=0)]

            b = b.T

            a = a[:, ~np.all(a[1:] == a[:-1], axis=0)]

            a = a.T

            b = b[:, ~np.all(a[1:] == a[:-1], axis=0)]

            b = b.T

            Array[i,:,:,0] = cv2.resize(b , (64,64))



        del a, b, df_train

        gc.collect()



        Temp = np.concatenate((Temp, Array), axis = 0)



        Temp_Label = np.concatenate((Temp_Label, Label.values), axis = 0)

        

    return Temp, Temp_Label

Temp, Temp_Label = Train_Data_Prepration()

Temp = Temp.astype('uint8')
Final_data1 = np.concatenate((Temp, Temp, Temp), axis = 3)
df_y = pd.DataFrame(Temp_Label, columns = ['image_id'])



df_y_new = pd.merge(df_y, df_train_csv, on = 'image_id', how = 'left')



y_train_168 = df_y_new.grapheme_root



y_train_11 = df_y_new.vowel_diacritic

y_train_7 = df_y_new.consonant_diacritic

df_y_new.to_csv("Label.csv")
df_y_new.head()
np.save('Y_train.npy', Temp_Label)

np.save('X_train2_3_Channel.npy', Final_data1)
