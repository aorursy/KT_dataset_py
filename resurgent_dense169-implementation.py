import os

from tqdm.auto import tqdm

from keras import layers,optimizers

import numpy as np

import pandas as pd

import gc

from matplotlib import pyplot as plt

%matplotlib inline

import cv2



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, Input, load_model

from keras.layers import Dense, Conv2D, Flatten, Activation, Concatenate

from keras.layers import MaxPool2D, AveragePooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras.initializers import RandomNormal

from keras.applications import DenseNet169

from sklearn.model_selection import train_test_split

from keras.layers import LeakyReLU

from skimage import exposure
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DIR = '../input/bengaliai-cv19'
train_df = pd.read_csv(os.path.join(DIR,'train.csv'))

test_df = pd.read_csv(os.path.join(DIR,'test.csv'))

class_map_df = pd.read_csv(os.path.join(DIR,'class_map.csv'))

sample_sub_df = pd.read_csv(os.path.join(DIR,'sample_submission.csv'))

                            

img_df = pd.read_parquet(os.path.join(DIR,'train_image_data_0.parquet'))
print(train_df.shape)

train_df.head()
print(test_df.shape)

test_df.head()
tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']
desc_df = train_df[tgt_cols].astype('str').describe()

desc_df
desc_df = train_df[tgt_cols].astype('str').describe()

desc_df
types = desc_df.loc['unique',:]
SIZE = 64    

N_ch = 1
def build_densenet():

    

    

    densenet = DenseNet169(weights='imagenet', include_top=False)



    input = Input(shape=(SIZE, SIZE, N_ch))

    x = Conv2D(3, (3, 3), padding='same')(input)

    

    x = densenet(x)

    

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(256)(x)

    x= LeakyReLU(alpha=0.1)(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)



    # multi output

    grapheme_root = Dense(types['grapheme_root'],

                          activation = 'softmax', name='root')(x)

    vowel_diacritic = Dense(types['vowel_diacritic'],

                            activation = 'softmax', name='vowel')(x)

    consonant_diacritic = Dense(types['consonant_diacritic'],

                                activation = 'softmax', name='consonant')(x)



    # model

    model = Model(input,

                  [grapheme_root, vowel_diacritic, consonant_diacritic])

    

    return model
model = build_densenet()

    

    

model.compile(Adam(lr=0.002),

              loss={'root': 'categorical_crossentropy',

                    'vowel': 'categorical_crossentropy',

                    'consonant': 'categorical_crossentropy'},

              metrics={'root': 'accuracy',

                       'vowel': 'accuracy',

                       'consonant': 'accuracy'}

             )

model.summary()
def AHE(img):

    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    return img_adapteq
def resize(df, size=64):

    resized = {}

    for i in range(df.shape[0]):

        img = AHE(df.loc[df.index[i]].values.reshape(137,236))

        image = cv2.resize(img,(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized_df = pd.DataFrame(resized).T

    return resized_df
img_df = img_df.drop(['image_id'], axis = 1)

X_df = (resize(img_df, SIZE) / 255.).astype('float32')

del img_df

gc.collect()

for i in tqdm(range(1,4)):

    img_df = pd.read_parquet(os.path.join(

    DIR, 'train_image_data_'+str(i)+'.parquet'))

    img_df = img_df.drop(['image_id'], axis = 1)

    img_df = (resize(img_df, SIZE) / 255.).astype('float32')

    X_df = pd.concat([X_df, img_df], axis = 0)

    del img_df

    gc.collect()

    

X_train = X_df.values.reshape(-1, SIZE, SIZE, N_ch)

del X_df

gc.collect()
train_df = train_df[tgt_cols].astype('uint8')

for col in tgt_cols:

    train_df[col] = train_df[col].map('{:03}'.format)

Y_train = pd.get_dummies(train_df)



del train_df

gc.collect()
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train,

                                                test_size=0.1, random_state=42)

y_train_root = y_train.iloc[:,0:types['grapheme_root']]

y_train_vowel = y_train.iloc[:,types['grapheme_root']:types['grapheme_root']+types['vowel_diacritic']]

y_train_consonant = y_train.iloc[:,types['grapheme_root']+types['vowel_diacritic']:]

y_test_root = y_test.iloc[:,0:types['grapheme_root']]

y_test_vowel = y_test.iloc[:,types['grapheme_root']:types['grapheme_root']+types['vowel_diacritic']]

y_test_consonant = y_test.iloc[:,types['grapheme_root']+types['vowel_diacritic']:]

    

del X_train, Y_train

gc.collect()
batch_size = 128

epochs = 25
reduceLR = ReduceLROnPlateau(monitor = 'val_root_loss',

                             patience = 2,

                             factor = 0.5,

                             min_lr = 1e-5,

                             verbose = 1)



chkPoint = ModelCheckpoint('dense169.h5',

                           monitor = 'val_root_accuracy',

                           save_best_only = True,

                           save_weights_only = False,

                           mode = 'auto',

                           period = 1,

                           verbose = 0)



earlyStop = EarlyStopping(monitor='val_root_accuracy',

                          mode = 'auto',

                          patience = 3,

                          min_delta = 0,

                          verbose = 1)
H = model.fit(x_train,

                    {'root': y_train_root,

                     'vowel': y_train_vowel,

                     'consonant': y_train_consonant},

                    batch_size=batch_size,

                    epochs =epochs,

                    shuffle = True,

                    validation_data = (x_test,

                                       {'root': y_test_root,

                                        'vowel': y_test_vowel,

                                        'consonant': y_test_consonant}),

                    callbacks = [reduceLR, chkPoint, earlyStop],

                    verbose = 1)



del x_train, x_test, y_train, y_test

gc.collect()



row_ids = []

targets = []      # prediction result

id = 0

for i in range(4):

    img_df = pd.read_parquet(os.path.join(

                            DIR, 'test_image_data_'+str(i)+'.parquet'))

    img_df = img_df.drop('image_id', axis = 1)

    img_df = resize(img_df, SIZE) / 255.

    X_test = img_df.values.reshape(-1, SIZE, SIZE, N_ch)



    preds = model.predict(X_test)

    for j in range(len(X_test)):

        for k in range(3):

            row_ids.append('Test_'+str(id)+'_'+tgt_cols[k])

            targets.append(np.argmax(preds[k][j]))

        id += 1
submit_df = pd.DataFrame({'row_id':row_ids,'target':targets},

                         columns = ['row_id','target'])

submit_df.head(10)
submit_df.to_csv('submission.csv',index=False)