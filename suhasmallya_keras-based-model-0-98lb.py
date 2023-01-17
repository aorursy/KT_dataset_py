import gc, os, sys, time

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import cv2
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn import metrics

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
!cd ../input; ls
df_train = pd.read_csv('../input/train_labels.csv')
df_test = pd.read_csv('../input/sample_submission.csv')
print(df_train.head())
print(df_test.head())
x_train = []
y_train = []
x_test = []
for name, tag in tqdm(df_train.values, miniters=300):
    #print(name)
    img = cv2.imread('../input/train/{}.jpg'.format(name))
    x_train.append(cv2.resize(img, (128, 128)))
    y_train.append(tag)
    
for name, tag in tqdm(df_test.values, miniters=300):
    #print(name)
    img = cv2.imread('../input/test/{}.jpg'.format(int(name)))
    x_test.append(cv2.resize(img, (128, 128)))
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)/255.
x_test  = np.array(x_test, np.float32)/255.

print(x_train.shape)
print(y_train.shape)

num_folds = 5
count_folds = 0

sum_score = 0

yfull_test = []
yfull_train = []

kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)
eval_func = metrics.roc_auc_score
def model_func():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(128, 128, 3)))
    # Without BatchNormalization
    # model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = optimizers.Adam(lr = 0.001, decay = 1e-6)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
    #print(model.summary())
    return model
def run_net(train_x, train_y, test_x, kf):      
    preds_train = np.zeros(len(train_x), dtype = np.float)
    preds_test = np.zeros(len(test_x), dtype = np.float)
    train_loss = []; test_loss = []

    i = 1
    
    for train_index, test_index in kf.split(train_x):
        start_time = time.time()
        x_tr = train_x[train_index]; x_te = train_x[test_index]
        y_tr = train_y[train_index]; y_te = train_y[test_index]

        datagen = ImageDataGenerator(
            rotation_range = 30,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        datagen.fit(x_tr)

        model = model_func()
        earlystop = EarlyStopping(monitor='val_loss', patience = 15, verbose=0, mode='auto')
        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 160),
            validation_data = (x_te, y_te), callbacks = [earlystop],
            steps_per_epoch = len(x_train) / 160, epochs = 100, verbose = 2)

        train_loss.append(eval_func(y_tr, model.predict(x_tr)[:, 0]))
        test_loss.append(eval_func(y_te, model.predict(x_te)[:, 0]))

        preds_train[test_index] = model.predict(x_te)[:, 0]
        preds_test += model.predict(test_x)[:, 0]

        print('KFold {0}: Train: {1:0.5f} Validation: {2:0.5f}'.format(i, train_loss[-1], test_loss[-1]))
        i += 1

    preds_test /= num_folds
    return preds_train, preds_test
train_predictions, test_predictions = run_net(x_train, y_train, x_test, kf)
df_test['invasive'] = test_predictions
df_test.to_csv('submission_v2.csv', index=False)