# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet201
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
df_train = pd.read_csv('../input/super-ai-image-classification/train/train/train.csv')
df_train['path'] = df_train['id'].apply(lambda x : '../input/super-ai-image-classification/train/train/images/'+x)
df_train
def get_img(path, size):
    return cv2.resize(cv2.imread(path), size)
plt.imshow(get_img('../input/super-ai-image-classification/train/train/images/6380fb87-18fe-4b76-a085-639a4e01b664.jpg',(300,300)))
img_size = (300,300)
x_train = np.empty((df_train.shape[0], img_size[0], img_size[1], 3), dtype=np.uint8)
all_path = df_train['path']
for i in range(all_path.shape[0]):
    x_train[i] = get_img(all_path[i], img_size)
print(x_train.shape)
x_train
y_train = np.array(pd.get_dummies(df_train['category']), dtype=np.uint8)
y_train
n_epoch = 2
fold_accuracy = []
fold_loss = []
kf = KFold(n_splits = 10)
c = 1

base_model = DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0],img_size[1],3)
)
GAP_layer = layers.GlobalAveragePooling2D()
drop_layer = layers.Dropout(0.6)
dense_layer = layers.Dense(2, activation='sigmoid', name='final_output')
    
final_output = dense_layer( drop_layer( GAP_layer(base_model.layers[-1].output) ) )
model = Model(base_model.layers[0].input, final_output)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.000005),
    metrics=['accuracy']
)

for train, test in kf.split(x_train, y_train):

    print('fold : '+ str(c))

    model.fit(x_train[train], y_train[train],
              batch_size=32,
              epochs=n_epoch
             )
    
    scores = model.evaluate(x_train[test], y_train[test])
    print('Score --> loss :' + str(scores[0]) + ', accuracy :' + str(scores[1]*100) + '%')
    fold_accuracy.append(scores[1] * 100)
    fold_loss.append(scores[0])

    c += 1
df_val = pd.read_csv('./val.csv')
df_val['path'] = df_val['id'].apply(lambda x : '../input/super-ai-image-classification/val/val/images/'+x)
df_val
plt.imshow(get_img('../input/super-ai-image-classification/val/val/images/a228f8f3-e525-4b1d-9374-af1cf858ab40.jpg', img_size))
x_val = np.empty((df_val.shape[0], img_size[0], img_size[1], 3), dtype=np.uint8)
all_path = df_val['path']
for i in range(all_path.shape[0]):
    x_val[i] = get_img(all_path[i], img_size)
x_val
predictions = model.predict(x_val).argmax(axis=-1)
predictions
df_val['category'] = predictions
df_val
df_val[['id', 'category']].to_csv("val.csv", index=False)
models = []
models.append(model)
models