import h5py

import os

import matplotlib.pyplot as plt

import numpy as np

import keras

from keras.utils.io_utils import HDF5Matrix

base_path = os.path.join('..', 'input')

train_h5_path = os.path.join(base_path, 'food_c101_n10099_r64x64x3.h5')

test_h5_path = os.path.join(base_path, 'food_test_c101_n1000_r64x64x3.h5')

%matplotlib inline
sample_imgs = 4

with h5py.File(train_h5_path, 'r') as n_file:

    label_names = [x.decode() for x in n_file['category_names'].value]

    hot_dog_idx = np.where(np.array(label_names)=='hot_dog')[0][0]

    is_hot_dog = (np.argmax(n_file['category'],1) == hot_dog_idx)

    hot_dog_ids = np.where(is_hot_dog)[0]

    not_hot_dog_ids = np.where(~is_hot_dog)[0]

    print('Total Hot Dogs:', len(hot_dog_ids))

    img_list = np.concatenate([np.random.choice(hot_dog_ids, sample_imgs), 

                    np.random.choice(not_hot_dog_ids, sample_imgs)])

    fig, m_ax = plt.subplots(2, 4, figsize = (12, 6))

    for c_ax, c_idx in zip(m_ax.flatten(), img_list):

        c_img = n_file['images'][c_idx]

        c_label = n_file['category'][c_idx]

        c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

        c_ax.axis('off')

        c_ax.set_title(label_names[np.argmax(c_label)])
train_list = np.concatenate([hot_dog_ids, 

                    np.random.choice(not_hot_dog_ids, hot_dog_ids.shape[0])]) # 50/50 split



X_train = HDF5Matrix(train_h5_path, 'images')[:][train_list]

y_train = HDF5Matrix(train_h5_path, 'category')[:]



y_train_is_hot_dog = (np.argmax(y_train[train_list],-1) == hot_dog_idx)



print('In Data',X_train.shape,'=>', y_train_is_hot_dog.shape, 'mean hotdog', y_train_is_hot_dog.mean())
y_test = HDF5Matrix(test_h5_path, 'category')[:]

is_hot_dog = (np.argmax(y_test,1) == hot_dog_idx)

hot_dog_ids = np.where(is_hot_dog)[0]

not_hot_dog_ids = np.where(~is_hot_dog)[0]

print('Test set has',len(hot_dog_ids), 'hot dogs')

test_list = np.concatenate([hot_dog_ids, 

                    np.random.choice(not_hot_dog_ids, hot_dog_ids.shape[0])]) # 50/50 split

X_test = HDF5Matrix(test_h5_path, 'images')[:][test_list]

y_test_is_hot_dog = (np.argmax(y_test[test_list],-1) == hot_dog_idx)

print('In Data',X_test.shape,'=>', y_test_is_hot_dog.shape)
from tpot import TPOTRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import FunctionTransformer, Normalizer

from sklearn.pipeline import make_pipeline

use_tpot = False

full_pipeline = make_pipeline(

    FunctionTransformer(lambda x: x.reshape((x.shape[0],-1)), validate = False), 

    Normalizer(),

    TPOTClassifier(generations=1, population_size=3, verbosity=2, random_state = 1234,

                  max_eval_time_mins=0.1) if use_tpot else RandomForestClassifier()

)
%%time

full_pipeline.fit(X_train, y_train_is_hot_dog)
%%time

y_train_pred = full_pipeline.predict(X_train)

y_pred = full_pipeline.predict(X_test)

print('Training hot dog likelihood',y_train_pred.mean(), 'Testing hot dog likelihood',y_pred.mean())
from sklearn.metrics import classification_report, classification, accuracy_score

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

ax1.matshow(classification.confusion_matrix(y_train_is_hot_dog, y_train_pred))

ax1.set_title('Training Results')

vmat = classification.confusion_matrix(y_test_is_hot_dog, y_pred)

ax2.matshow(vmat)

ax2.set_title('Validation Results')

print('Validation Accuracy: %2.1f%%' % (100*accuracy_score(y_test_is_hot_dog, y_pred)))

print(vmat)
print(classification_report(y_test_is_hot_dog, y_pred))
sample_imgs = 4

with h5py.File(test_h5_path, 'r') as n_file:

    label_names = [x.decode() for x in n_file['category_names'].value]

    hot_dog_idx = np.where(np.array(label_names)=='hot_dog')[0][0]

    is_hot_dog = (np.argmax(n_file['category'],1) == hot_dog_idx)

    hot_dog_ids = np.where(is_hot_dog)[0]

    not_hot_dog_ids = np.where(~is_hot_dog)[0]

    print('Total Hot Dogs:', len(hot_dog_ids))

    img_list = np.concatenate([np.random.choice(hot_dog_ids, sample_imgs), 

                    np.random.choice(not_hot_dog_ids, sample_imgs)])

    fig, m_ax = plt.subplots(2, 4, figsize = (12, 6))

    for c_ax, c_idx in zip(m_ax.flatten(), img_list):

        c_img = n_file['images'][c_idx]

        c_pred = full_pipeline.predict_proba(np.expand_dims(c_img,0))[0]

        c_label = n_file['category'][c_idx]

        c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

        c_ax.axis('off')

        c_ax.set_title('Hot Dog Likelihood: %2.1f%%\nActual: %s' % (c_pred[1]*100,

                                                  label_names[np.argmax(c_label)]))