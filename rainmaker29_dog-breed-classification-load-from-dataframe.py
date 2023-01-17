# !pip install gdown
# !gdown --id 1gsN1uvjJ-aruYqbi-68of5MccsKclBgH

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gsN1uvjJ-aruYqbi-68of5MccsKclBgH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gsN1uvjJ-aruYqbi-68of5MccsKclBgH" -O dog_breed_classification_ai_challenge-dataset.zip && rm -rf /tmp/cookies.txt

!unzip dog_breed_classification_ai_challenge-dataset.zip
!rm ./dog_breed_classification_ai_challenge-dataset.zip
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time

import os
labels_dataframe = pd.read_csv('/kaggle/working/dataset/train.csv')
sample = pd.DataFrame()
sf = []
for x in os.listdir("/kaggle/working/dataset/test"):
    sf.append(x)

sample['Filename']=sf
    
ix = np.random.permutation(len(labels_dataframe))
labels_dataframe = labels_dataframe.iloc[ix]
class_to_index = dict({breed:ix for ix, breed in enumerate(labels_dataframe['Labels'].unique())})
index_to_class = dict({ix:breed for ix, breed in enumerate(labels_dataframe['Labels'].unique())})
labels_dataframe['Labels'] = labels_dataframe['Labels'].map(class_to_index)
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
            path = os.path.join(img_dir, filename)
            image = load_img(path, target_size = (image_shape[0], image_shape[1]))
            image = img_to_array(image)
            images[ix] = image
            labels[ix] = dataframe.loc[ix, y_col]
        print('Found {} validated image filenames belonging to {} classes.'.format(no_of_images, np.unique(labels).size))
        return images, labels
    else:
        for ix in range(no_of_images):
            filename = dataframe.loc[ix, x_col]
            path = os.path.join(img_dir, filename)
            image = load_img(path, target_size = (image_shape[0], image_shape[1]))
            image = img_to_array(image)
            images[ix] = image
        print('Found {} validated image filenames'.format(no_of_images))
        return images
params = dict(dataframe = labels_dataframe, image_shape = image_shape, img_dir = '/kaggle/working/dataset/train', x_col = 'Filename', y_col = 'Labels')
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
import tensorflow as tf
import tensorflow.keras.backend as K

def f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
from keras.layers import Dropout, Dense

def create_model(features_shape = 1024):
    model = Sequential()
    model.add(InputLayer(input_shape = (features_shape, )))
    model.add(Dropout(0.6))
    model.add(Dense(8192, activation = 'relu'))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='relu')) 
    model.add(Dropout(0.6))
    model.add(Dense(len(class_to_index), activation = 'softmax'))
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer ='SGD', metrics = ['accuracy'])
    return model
model = create_model(final_features.shape[1])
model.summary()
model.fit(final_features, labels, batch_size = 512, epochs = 50) 
model.save('dogspretrain.h5')
params = dict(dataframe = sample, image_shape = image_shape, img_dir = '/kaggle/working/dataset/test', x_col = 'Filename')
images = run_with_timer(load_from_dataframe, params)
inception_feature = run_with_timer(get_feature, dict(model_name = InceptionV3, preprocess_input = inception_preprocess, images = images, pooling = 'avg'))
nasnet_features = run_with_timer(get_feature, dict(model_name = NASNetLarge, preprocess_input = nasnet_preprocessor, images = images, pooling = 'avg'))
xception_feature = run_with_timer(get_feature, dict(model_name = Xception, preprocess_input = xception_preprocess, images = images, pooling = 'avg'))
resnet_feature = run_with_timer(get_feature, dict(model_name = InceptionResNetV2, preprocess_input = resnet_preprocess, images = images, pooling = 'avg'))
final_features = np.concatenate([inception_feature, nasnet_features, xception_feature, resnet_feature], axis = 1)
print('final features shape: {}'.format(final_features.shape))
del images, inception_feature, nasnet_features, xception_feature, resnet_feature
prediction = model.predict(final_features)
submission = pd.DataFrame({'Filename':sample.Filename.values})
submission['Filename'] = submission['Filename'].apply(lambda x : x.split('.')[0])
prediction = pd.DataFrame(prediction)
prediction.columns = class_to_index.keys()
submission = pd.concat([submission, prediction], axis = 1)
submission.to_csv('submission.csv', index = False)
submission.loc[submission['Filename']=='1_test',:]
# index_to_class[np.argmax(np.array(submission.iloc[946][1:].to_list()))]
submission.iloc[0][0]
mysub = pd.DataFrame()
files=[]
for x in os.listdir('/kaggle/working/dataset/test'):
    files.append(x)
    
mysub['Filename']= files
mysub['intfile'] = mysub['Filename'].apply(lambda x : int(x.split('_')[0]))
mysub
files=[]
res=[]
for i in range(len(submission)):
    files.append(submission.iloc[i][0]+'.jpg')
    res.append(index_to_class[np.argmax(np.array(submission.iloc[i][1:].to_list()))])
    
    
    
findf = pd.DataFrame()
findf['Filename'] = files
findf['Labels'] = res
findf['intfile'] = findf['Filename'].apply(lambda x : int(x.split("_")[0]))
findf=findf.sort_values(by=['intfile'])
findf = findf[['Filename','Labels']]
findf.to_csv('output.csv',header=True,index=False)
findf