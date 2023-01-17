import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
labels = pd.read_csv('../input/dog-breed-identification/labels.csv')

labels.head()
top_breeds = sorted(list(labels['breed'].value_counts().head(16).index))

labels = labels[labels['breed'].isin(top_breeds)]
len(labels.breed.value_counts())
np.average(labels.breed.value_counts())
labels.breed.value_counts().plot(kind='bar')
train = labels.copy()

train['filename'] = train.apply(lambda x: ('../input/dog-breed-identification/train/' + x['id'] + '.jpg'), axis=1)

train.head()
from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from sklearn.model_selection import train_test_split
train_data = np.array([ img_to_array(load_img(img, target_size=(299, 299))) for img in train['filename'].values.tolist()]).astype('float32')
train_data.shape
labels = train.breed

labels
x_train, x_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.2)
x_train.shape, y_train.shape, x_val.shape, y_val.shape
y_train = pd.get_dummies(y_train.reset_index()).values[:,1:]
y_val = pd.get_dummies(y_val.reset_index()).values[:, 1:]
x_train.shape, y_train.shape, x_val.shape, y_val.shape
from os import makedirs

from os.path import expanduser, exists, join



!ls ../input/keras-pretrained-models/



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

    

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
import keras

from keras.applications import Xception, InceptionV3

from keras.applications.xception import preprocess_input as xception_preprocessor

from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.preprocessing.image import ImageDataGenerator
xception_model = Xception(include_top=False, input_shape=(299, 299, 3), pooling='avg')

inception_model = InceptionV3(include_top=False, input_shape=(299, 299, 3), pooling='avg')
train_generator = ImageDataGenerator(zoom_range = 0.3, width_shift_range=0.1, height_shift_range=0.1)

val_generator = ImageDataGenerator()
train_generator.preprocessing_function = inception_v3_preprocessor

val_generator.preprocessing_function = inception_v3_preprocessor



generator = train_generator.flow(x_train, y_train, shuffle=False)

inception_train_bottleneck = inception_model.predict_generator(generator, verbose=1)

np.save('inception-train.npy', inception_train_bottleneck)



generator = val_generator.flow(x_val, y_val, shuffle=False)

inception_val_bottleneck = inception_model.predict_generator(generator, verbose=1)

np.save('inception-val.npy', inception_val_bottleneck)
train_generator.preprocessing_function = xception_preprocessor

val_generator.preprocessing_function = xception_preprocessor



generator = train_generator.flow(x_train, y_train, shuffle=False)

xception_train_bottleneck = xception_model.predict_generator(generator, verbose=1)

np.save('xception-train.npy', xception_train_bottleneck)



generator = val_generator.flow(x_val, y_val, shuffle=False)

xception_val_bottleneck = xception_model.predict_generator(generator, verbose=1)

np.save('xception-val.npy', xception_val_bottleneck)
xception_train_bottleneck_2 = np.load('../input/bottleneck-features-dog-breeds-identification/inception-train.npy')

xception_val_bottleneck_2 = np.load('../input/bottleneck-features-dog-breeds-identification/inception-val.npy')



inception_train_bottleneck_2 = np.load('../input/bottleneck-features-dog-breeds-identification/xception-train.npy')

inception_val_bottleneck_2 = np.load('../input/bottleneck-features-dog-breeds-identification/xception-val.npy')



xception_train_bottleneck_3 = np.load('../input/bottleneck-features-inceptionxception/Xception_features.npy')

xception_val_bottleneck_3 = np.load('../input/bottleneck-features-inceptionxception/Xception_validfeatures.npy')



inception_train_bottleneck_3 = np.load('../input/bottleneck-features-inceptionxception/InceptionV3_features.npy')

inception_val_bottleneck_3 = np.load('../input/bottleneck-features-inceptionxception/InceptionV3_validfeatures.npy')



xception_train_bottleneck_4 = np.load('./xception-train.npy')

xception_val_bottleneck_4 = np.load('./xception-val.npy')



inception_train_bottleneck_4 = np.load('./inception-train.npy')

inception_val_bottleneck_4 = np.load('./inception-val.npy')
xception_train_bottleneck.shape, xception_val_bottleneck.shape, inception_train_bottleneck.shape, inception_val_bottleneck.shape
xception_train_bottleneck_2.shape, xception_val_bottleneck_2.shape, inception_train_bottleneck_2.shape, inception_val_bottleneck_2.shape
xception_train_bottleneck.dtype, xception_val_bottleneck.dtype, inception_train_bottleneck.dtype, inception_val_bottleneck.dtype
xception_train_bottleneck_2.dtype, xception_val_bottleneck_2.dtype, inception_train_bottleneck_2.dtype, inception_val_bottleneck_2.dtype
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score



logreg_inception = LogisticRegression(multi_class='multinomial', solver='lbfgs')

logreg_inception.fit(inception_train_bottleneck_4, (y_train * range(16)).sum(axis=1))

inception_preds_val = logreg_inception.predict(inception_val_bottleneck_4)

inception_preds_train = logreg_inception.predict(inception_train_bottleneck_4)



logreg_xception = LogisticRegression(multi_class='multinomial', solver='lbfgs')

logreg_xception.fit(xception_train_bottleneck_4, (y_train * range(16)).sum(axis=1))

xception_preds_val = logreg_xception.predict(xception_val_bottleneck_4)

xception_preds_train = logreg_xception.predict(xception_train_bottleneck_4)
avgpreds_val = np.average([inception_preds_val, xception_preds_val], axis=0, weights=[1,1])

avgpreds_train = np.average([inception_preds_train, xception_preds_train], axis=0, weights=[1,1])

avgpreds_val.shape, avgpreds_train.shape
accuracy_score(np.round(avgpreds_val).astype('int'), np.argmax(y_val, axis=1))
accuracy_score(np.round(avgpreds_train).astype('int'), np.argmax(y_train, axis=1))
import tensorflow as tf



with tf.Session() as sess:

    result = sess.run(tf.one_hot(np.round(avgpreds_val), depth = 16))

    print('ensemble validation accuracy : {}'.format(accuracy_score(y_val, result)))