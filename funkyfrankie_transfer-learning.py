from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.applications.vgg16 import VGG16, preprocess_input
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
import time
import os

VGG16_PATH = '../input/vgg16/'
DOG_PATH = '../input/dog-breed-identification/'
BATCH_SIZE = 50
def path_by_id(id, test=False):
    return DOG_PATH + ('train/' if not test else 'test/') + id + '.jpg'

def get_image_array(id, test=False):
    img = image.load_img(path_by_id(id, test), target_size=(224, 224))
    return image.img_to_array(img)
labels_data = pd.read_csv(DOG_PATH + 'labels.csv')
labels = labels_data.set_index('id')['breed']
y = pd.get_dummies(labels, sparse = True)
train_ids = y.index
test_ids = pd.read_csv(DOG_PATH + 'sample_submission.csv')['id']
# Instantiate the model with the pre-trained weights (no top)
base_model= VGG16(weights=(VGG16_PATH+'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                  include_top=False, pooling='avg')
def process_image(id, test=False):
    return base_model.predict(preprocess_input(np.expand_dims(get_image_array(id, test), axis=0)))
X_arr = []
for id in tqdm(train_ids):
    X_arr.append(process_image(id)[0])
X = pd.DataFrame(X_arr, index=train_ids)
# model = XGBClassifier(objective='multi:softmax', num_class=120, n_jobs=4, verbose=True)
# model = LogisticRegression(n_jobs=4, verbose=True)
model = Sequential([
    Dense(1024, input_shape=(512,)),
    Activation('relu'),
    Dense(256, input_shape=(512,)),
    Activation('relu'),
    Dense(120),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = X
y_train = y
model.fit(X_train, np.asarray(y_train), epochs=100, batch_size=100, verbose=False)
# model.fit(batch_iterator(range(0,100)),
#         samples_per_epoch=1, nb_epoch=100)

X_test = []
for id in tqdm(test_ids):
    X_test.append(process_image(id, test=True)[0])
X_test = np.array(X_test)
# y_pred_cv = pd.DataFrame(model.predict(X_cv))
# y_cv = (np.asarray(y_cv) * range(120)).sum(axis=1)
# y_pred_cv = y_pred_cv.idxmax(axis=1)
# (y_cv== np.asarray(y_pred_cv)).sum() / len(y_cv)
y_pred = pd.DataFrame(model.predict(X_test), index=test_ids)
y_pred = y_pred.idxmax(axis=1)
y_pred = pd.get_dummies(y_pred, prefix='', prefix_sep='')
y_pred.columns = y_pred.columns.astype('int')
missing_cols = {x for x in range(120)} - set(y_pred.columns)
for col in missing_cols:
    y_pred[col] = 0
y_pred = y_pred.reindex_axis(sorted(y_pred.columns), axis=1)
y_pred.columns = y.columns
y_pred
y_pred.to_csv('submission.csv', index_label='id')