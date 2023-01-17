import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import time



import os
labels_dataframe = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')

sample = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
ix = np.random.permutation(len(labels_dataframe))

labels_dataframe = labels_dataframe.iloc[ix]
class_to_index = dict({breed:ix for ix, breed in enumerate(labels_dataframe['breed'].unique())})

index_to_class = dict({ix:breed for ix, breed in enumerate(labels_dataframe['breed'].unique())})

labels_dataframe['breed'] = labels_dataframe['breed'].map(class_to_index)
image_shape = (331, 331, 3)
st_time = 0

def start_timer():

    global st_time

    st_time = time.time()

def stop_timer():

    global st_time

    total = time.time() - st_time

    st_time = 0

    print('total runtime: {}'.format(total))



def run_with_timer(function, param, return_value = True):

    if return_value == True:

        start_timer()

        result = function(**param)

        stop_timer()

        return result

    else:

        start_timer()

        function(**param)

        stop_timer()

    
def load_from_dataframe(dataframe, image_shape, img_dir, x_col = None, y_col = None,):

    no_of_images = len(dataframe)

    images = np.zeros((no_of_images, image_shape[0], image_shape[1], image_shape[2]), dtype = np.uint8)

    if y_col:

        labels = np.zeros((no_of_images, 1), dtype = np.uint8)

        for ix in range(no_of_images):

            filename = dataframe.loc[ix, x_col]

            path = os.path.join(img_dir, filename + '.jpg')

            image = load_img(path, target_size = (image_shape[0], image_shape[1]))

            image = img_to_array(image)

            images[ix] = image

            labels[ix] = dataframe.loc[ix, y_col]

        print('Found {} validated image filenames belonging to {} classes.'.format(no_of_images, np.unique(labels).size))

        return images, labels

    else:

        for ix in range(no_of_images):

            filename = dataframe.loc[ix, x_col]

            path = os.path.join(img_dir, filename + '.jpg')

            image = load_img(path, target_size = (image_shape[0], image_shape[1]))

            image = img_to_array(image)

            images[ix] = image

        print('Found {} validated image filenames'.format(no_of_images))

        return images
params = dict(dataframe = labels_dataframe, image_shape = image_shape, img_dir = '/kaggle/input/dog-breed-identification/train', x_col = 'id', y_col = 'breed')

images, labels = run_with_timer(load_from_dataframe, params)
from keras import Sequential

from keras.layers import Lambda, InputLayer



def get_feature(model_name, preprocess_input, images, pooling = 'avg', target_size = (331,331,3)):

    base_model = model_name(input_shape = target_size, include_top=False, pooling = pooling)



    model = Sequential()

    model.add(InputLayer(input_shape = target_size))

    model.add(Lambda(preprocess_input))

    model.add(base_model)



    feature = model.predict(images)

    

    print('feature-map shape: {}'.format(feature.shape))

    return feature
from keras.applications.inception_v3 import InceptionV3, preprocess_input



inception_preprocess = preprocess_input

params = dict(model_name = InceptionV3, preprocess_input = inception_preprocess, images = images, pooling = 'avg')

inception_feature = run_with_timer(get_feature, params)
from keras.applications.nasnet import NASNetLarge, preprocess_input



nasnet_preprocessor = preprocess_input

params = dict(model_name = NASNetLarge, preprocess_input = nasnet_preprocessor, images = images, pooling = 'avg')

nasnet_features = run_with_timer(get_feature, params)
from keras.applications.xception import Xception, preprocess_input



xception_preprocess = preprocess_input

params = dict(model_name = Xception, preprocess_input = xception_preprocess, images = images, pooling = 'avg')

xception_feature = run_with_timer(get_feature, params)
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input



resnet_preprocess = preprocess_input

params = dict(model_name = InceptionResNetV2, preprocess_input = resnet_preprocess, images = images, pooling = 'avg')

resnet_feature = run_with_timer(get_feature, params)
final_features = np.concatenate([inception_feature, nasnet_features, xception_feature, resnet_feature], axis = 1)

print('final features shape: {}'.format(final_features.shape))

del images, inception_feature, nasnet_features, xception_feature, resnet_feature
from keras.layers import Dropout, Dense



def create_model(features_shape = 1024):

    model = Sequential()

    model.add(InputLayer(input_shape = (features_shape, )))

    model.add(Dropout(0.6))

    model.add(Dense(4096, activation = 'relu'))

    model.add(Dropout(0.6))

    model.add(Dense(len(class_to_index), activation = 'softmax'))

    

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer ='Adam', metrics = ['accuracy'])

    return model
model = create_model(final_features.shape[1])

model.summary()
model.fit(final_features, labels, batch_size = 128, epochs = 6, validation_split = 0.2)
params = dict(dataframe = sample, image_shape = image_shape, img_dir = '/kaggle/input/dog-breed-identification/test', x_col = 'id')

images = run_with_timer(load_from_dataframe, params)
inception_feature = run_with_timer(get_feature, dict(model_name = InceptionV3, preprocess_input = inception_preprocess, images = images, pooling = 'avg'))

nasnet_features = run_with_timer(get_feature, dict(model_name = NASNetLarge, preprocess_input = nasnet_preprocessor, images = images, pooling = 'avg'))

xception_feature = run_with_timer(get_feature, dict(model_name = Xception, preprocess_input = xception_preprocess, images = images, pooling = 'avg'))

resnet_feature = run_with_timer(get_feature, dict(model_name = InceptionResNetV2, preprocess_input = resnet_preprocess, images = images, pooling = 'avg'))
final_features = np.concatenate([inception_feature, nasnet_features, xception_feature, resnet_feature], axis = 1)

print('final features shape: {}'.format(final_features.shape))

del images, inception_feature, nasnet_features, xception_feature, resnet_feature
prediction = model.predict(final_features)

submission = pd.DataFrame({'id':sample.id.values})

submission['id'] = submission['id'].apply(lambda x : x.split('.')[0])

prediction = pd.DataFrame(prediction)

prediction.columns = class_to_index.keys()
submission = pd.concat([submission, prediction], axis = 1)

submission.to_csv('submission.csv', index = False)