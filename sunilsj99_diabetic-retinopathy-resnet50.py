# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tqdm import tqdm

import os

%matplotlib inline

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
train.head()
test.head()
path_train = '../input/aptos2019-blindness-detection/train_images/'

path_test = '../input/aptos2019-blindness-detection/test_images/'
def apply_text(fn):

    return fn+'.png'



def convert_str(fn):

    return str(fn)
train['id_code'] = train['id_code'].apply(apply_text)

train['diagnosis'] = train['diagnosis'].apply(convert_str)

test['id_code'] = test['id_code'].apply(apply_text)
train.head()
img = cv2.imread(path_train+train['id_code'][0])

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



plt.imshow(img)
from keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(rescale=1./255., validation_split=0.25, zoom_range=0.3, rotation_range=0.2)
train_generator = train_gen.flow_from_dataframe(train, directory=path_train, x_col='id_code', y_col='diagnosis', batch_size=32,

                                                subset="training", seed=42, target_size=(299,299))



valid_generator = train_gen.flow_from_dataframe(train, directory=path_train, x_col='id_code', y_col='diagnosis', batch_size=32,

                                                subset="validation", seed=42, target_size=(299,299))
test_gen = ImageDataGenerator(rescale=1./255.)
test_generator = test_gen.flow_from_dataframe(test, directory=path_test, x_col='id_code', y_col= None, batch_size=32,

                                                seed=42, target_size=(299,299), shuffle=False, class_mode=None)
sns.countplot(x=train['diagnosis'])
train.diagnosis.value_counts()
class_weight = {0: 1,

                1: 4.878,

                2: 1.806,

               3:9.352,

               4:6.118}
from keras import applications

from keras import optimizers, regularizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Activation

from keras import backend as k 

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint
model = applications.Xception(weights = None, include_top = False, input_shape = (299,299,3))
model.load_weights('../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
x = model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

predictions = Dense(5, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.summary()
es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr = 0.0001), metrics=["accuracy"])
history = model_final.fit_generator(train_generator, validation_data=valid_generator, class_weight = class_weight,

                                    steps_per_epoch = train_generator.n//train_generator.batch_size, validation_steps = valid_generator.n//valid_generator.batch_size,

                                   epochs = 10, callbacks=[es, mc])
# loss

plt.figure(figsize=(15,7))

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.legend()

plt.show()
plt.figure(figsize=(15,7))

plt.plot(history.history['acc'], label='train acc')

plt.plot(history.history['val_acc'], label='val acc')

plt.legend()

plt.show()
predictions = model_final.predict_generator(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = []



for i in predictions:

    predicted_classes.append(np.argmax(i))

    

predicted_classes[:5]
submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
submission.head()
submission['diagnosis'] = predicted_classes
submission.head()
submission.to_csv('submission.csv', index=False, sep=',')
def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return 1.0 - numerator / denominator