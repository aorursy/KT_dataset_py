# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import time, json, codecs

from sklearn.model_selection import train_test_split
batchsize=256

learning_rate=0.01

epochs=5

shape = (200, 136, 3)

debug=True

augment_data = False
t_start = time.time()
if not os.path.isdir('weights'):

    os.makedirs('weights')
def load_data():

    

    with codecs.open('/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/metadata.json','r',encoding='utf-8',errors='ignore') as f:

        train_meta = json.load(f)

    with codecs.open('/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/metadata.json','r',encoding='utf-8',errors='ignore') as f:

        test_meta = json.load(f)

        

    train_annotations = pd.DataFrame(train_meta['annotations'])

    categories = pd.DataFrame(train_meta['categories'])

    categories.columns = ['family', 'genus', 'category_id', 'category_name']

    train_images = pd.DataFrame(train_meta['images'])

    train_images.columns = ['file_name', 'height', 'image_id','license','width']

    

    X_test = pd.DataFrame(test_meta['images'])

    X_test.columns = ['file_name', 'height', 'image_id','license','width']

    X_test = X_test[['image_id','file_name']]

    

    regions = pd.DataFrame(train_meta['regions'])

    regions.columns=['region_id','name']

    

    Xorig = train_annotations.merge(categories,on='category_id', how="left"

                                     ).merge(train_images, on="image_id", how="outer"

                                            ).merge(regions, on="region_id", how="outer")

    X = Xorig[['file_name','family','genus','category_id']]

    

    name_list = X['family'].unique().tolist()

    X.loc[:,'family'] = X['family'].map(lambda x:name_list.index(x))

    genus_list = X['genus'].unique().tolist()

    X.loc[:,'genus'] = X['genus'].map(lambda x:genus_list.index(x))

    

    if debug:

        X=X[X['family']>290]

    return X.astype({'family':'int16','genus':'int16','category_id':'int16'}), X_test

X,X_test=load_data()

nmb_cat = X['category_id'].max()+1

nmb_gen = X['genus'].max()+1

nmb_fam = X['family'].max()+1

X_train, X_dev = train_test_split(X, test_size=0.05, shuffle=True, random_state=13)

del X
# first crop away the pathological margins of the images, and then apply the data augmentation (rotation, shifts, zooms)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt



gnrt1 = ImageDataGenerator(dtype='uint8')

gnrt2 = ImageDataGenerator(rotation_range=35,featurewise_center=False,

                                      featurewise_std_normalization=False,

                                      width_shift_range=0.1,

                                      height_shift_range=0.1,

                                      zoom_range=0.1,horizontal_flip=True,

                                      dtype='uint8')

    

def crop(batch_x):

    cut1 = int(0.1*batch_x.shape[1])

    cut2 = int(0.05*batch_x.shape[2])

    return batch_x[:,cut1:-cut1,cut2:-cut2]



def crop_generator(batches,test=False):

    while True:

        if test:

            batch_x = next(batches)

            yield next(gnrt2.flow(crop(batch_x),batch_size=batchsize))

        else:

            batch_x, batch_y = next(batches)

            yield (next(gnrt2.flow(crop(batch_x),batch_size=batchsize)), batch_y)

            

    

i=0

for x, y in crop_generator(gnrt1.flow_from_dataframe(

                                    dataframe=X_train[:2], directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                    x_col="file_name", y_col=['family','genus','category_id'], class_mode="multi_output",

                                    target_size=(shape[0],shape[1]),batch_size=batchsize,

                                    validate_filenames=False, verbose=False)):

    plt.imshow(x.astype('uint8')[1])

    i=i+1

    if i==1:

        break
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Input, concatenate, Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model



def create_model():

    actual_shape = (crop(np.zeros((1,shape[0],shape[1],shape[2]))).shape)[1:]

    i = Input(actual_shape)

    x = ResNet50(weights='imagenet', include_top=False, input_shape=actual_shape, pooling='max')(i)

    x = Flatten()(x)

    o1 = Dense(nmb_fam, name="family", activation='softmax')(x)

    o2 = concatenate([x,o1])

    o2 = Dense(nmb_gen, name="genus", activation='softmax')(o2)

    o3 = concatenate([x,o1,o2])

    o3 = Dense(nmb_cat, name="category_id", activation='softmax')(o3)

    model = Model(inputs=i,outputs=[o1,o2,o3])

    model.layers[1].trainable = False

    model.get_layer('genus').trainable = False

    model.get_layer('category_id').trainable = False

    return model
def compile(model,learning_rate=0.005):

    model.compile(optimizer=Adam(learning_rate=0.005),loss=["sparse_categorical_crossentropy",

                                     "sparse_categorical_crossentropy",

                                     "sparse_categorical_crossentropy"],

                                metrics=['accuracy'])

    



TRAINSTEPS = (X_train.shape[0]//batchsize)+1

VALSTEPS = (X_dev.shape[0]//batchsize)+1
def train(ep,initial_epoch=0):

    return model.fit_generator(gnrt1.flow_from_dataframe(

                                    dataframe=X_train, directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                    x_col="file_name", y_col=['family','genus','category_id'], class_mode="multi_output",

                                    target_size=(shape[0],shape[1]),batch_size=batchsize,

                                    validate_filenames=False, verbose=False),

                    validation_data=gnrt1.flow_from_dataframe(dataframe=X_dev, directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                    x_col="file_name", y_col=['family','genus','category_id'], class_mode="multi_output",

                                    target_size=(shape[0],shape[1]),batch_size=batchsize,

                                    validate_filenames=False, verbose=False),

                    epochs=ep+initial_epoch,max_queue_size=30, workers=16, #use_multiprocessing=True,

                               initial_epoch=initial_epoch,

                               steps_per_epoch=TRAINSTEPS,

                               validation_steps=VALSTEPS

                   )
model = create_model()

compile(model,learning_rate)

model.summary()

plot_model(model, show_shapes=True, show_layer_names=True)
t_before = time.time()

for i in range(epochs):

    hist = train(1,i)

    print("Time used for epoch {}: {} min".format(i+1,int((time.time()-t_before)/60)))

    gacc = hist.history['genus_accuracy'][0]

    facc = hist.history['family_accuracy'][0]

    if facc > 0.9:

        model.get_layer("family").trainable=False

        print("Stopped training family.")

        compile(model,learning_rate)

    if facc > 0.7:

        model.get_layer("genus").trainable=True

        print("Training genus layer now.")

        compile(model,learning_rate)

    if gacc > 0.9:

        model.get_layer("genus").trainable=False

        print("Stopped training genus.")

        compile(model,learning_rate)

    if gacc >0.7:

        model.get_layer("category_id").trainable = True

        print("Training category layer now.")

        compile(model,learning_rate)

filename="weights.h5"

model.save_weights(filename)

print("Saving weights to {}".format(filename))
def predict(X_test):

    STEPS_PREDICT = (X_test.shape[0]//batchsize)+1

    predictions = model.predict_generator(crop_generator(gnrt1.flow_from_dataframe(

                                    dataframe=X_test, directory='../input/herbarium-2020-fgvc7/nybg2020/test/',

                                    x_col="file_name",class_mode=None,

                                    target_size=(shape[0],shape[1]),batch_size=batchsize,

                                    validate_filenames=False, verbose=True),True),

                                     steps=STEPS_PREDICT, workers=8, use_multiprocessing=True)



    submission = pd.DataFrame()

    submission['Id'] = X_test['image_id']

    submission['Predicted'] = predictions[2].argmax(axis=1)

    submission.to_csv('submission.csv', index=False)

print("Submission file written. Total time elapsed: {} minutes".format((time.time()-t_start)//60))