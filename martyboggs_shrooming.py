# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)


# sklearn to do preprocessing & ML models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# keras for deep learning model creation
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import plot_model

# Matplotlob & seaborn to plot graphs & visulisation
import matplotlib.pyplot as plt 
import seaborn as sns

# for fixing the random seed
import random
import os, tensorflow as tf
import torch

# ignore warnings
# import warnings
# warnings.simplefilter(action='ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
shroom_data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
shroom_data.shape
shroom_data.head()
shroom_data.describe()
shroom_data.isna().sum()
shroom_data.dtypes
shroom_data['cap-shape'].value_counts()
sns.scatterplot(shroom_data['class'], shroom_data['cap-shape'])
target = shroom_data['class'].tolist()
shroom_data.drop(['class'], axis=1, inplace=True)
le = preprocessing.LabelEncoder()
target = le.fit_transform(target)
# print(*target)
shroom_data = pd.get_dummies(shroom_data, columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
shroom_data.head()

train_data, val_data, train_target, val_target = train_test_split(shroom_data, target, test_size=0.2)
train_data.shape, val_data.shape, len(train_target), len(val_target)
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

# We fix all the random seed so that, we can reproduce the results.
seed_everything(2020)
input_layer = Input(shape=(shroom_data.shape[1],),name='input_layer')
hidden_layer_1 = Dense(32, activation = 'relu')(input_layer)
hidden_layer_2 = Dense(16, activation = 'relu')(hidden_layer_1)
output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_2)

model = Model(input=input_layer, output=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)
# We will give training 10 times with the same data.
EPOCHS = 10

# We will process 64 rows at a time.
BATCH_SIZE = 32

model.fit(
        train_data, train_target,
        nb_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_data, val_target),
        verbose = 1,
)
# Predict labels on Validation data which model have never seen before.

val_predictions = model.predict(val_data)
len(val_predictions)
# convert to integers
val_predictions1 = [1 if x >= 0.5 else 0 for x in val_predictions]
val_predictions1[:10]
accuracy = accuracy_score(val_target, val_predictions1)
accuracy
print("We got %.3f percent accuracy on our validation unseen data !!"%(accuracy*100))