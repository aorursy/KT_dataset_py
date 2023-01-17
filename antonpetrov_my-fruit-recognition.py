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
!find '../input/fruit-recognition/' -type d -print
# Import all important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt
import random
import time
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.random import set_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import FBetaScore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from matplotlib.image import imread
np.random.seed(42)
%matplotlib inline

def load_images_from_folder(folder, only_path = False, label = ""):
    if only_path == False:
        images = []
        for filename in os.listdir(folder):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images
    else:
        path = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder,filename)
            if img_path is not None:
                path.append([label,img_path])
        return path

images = []
dirp = "../input/fruit-recognition/"
images = []

for f in os.listdir(dirp):
    if "png" in os.listdir(dirp+f)[0]:
        images += load_images_from_folder(dirp+f,True,label = f)
    else: 
        for d in os.listdir(dirp+f):
            images += load_images_from_folder(dirp+f+"/"+d,True,label = f)
df = pd.DataFrame(images, columns = ["fruit", "path"])
df.head()
df.loc[:,'fruit'].nunique()
fruit_names = sorted(df.fruit.unique())
mapped_fruit_names = dict(zip([t for t in range(len(fruit_names))], fruit_names))
df["label"] = df["fruit"].map(mapped_fruit_names)
fruit_values = df.loc[:,'fruit'].value_counts
print(mapped_fruit_names)
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(12, 8)})
plt.xticks(rotation=45)
sns.barplot(data=pd.DataFrame(fruit_values(normalize=False)).reset_index(), x='index', y='fruit', dodge=False)
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(12, 8)})
plt.xticks(rotation=45)
sns.barplot(data=pd.DataFrame(fruit_values(normalize=True)).reset_index(), x='index', y='fruit')
fruit_values
fruit_values(normalize=True)
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.path[i]))
    ax.set_title(df.fruit[i])
    
plt.tight_layout()
plt.show()
df.shape
strat_split = StratifiedShuffleSplit(n_splits=5, test_size=0.07, random_state=42)
for tr_index, val_index in strat_split.split(df, df.loc[:,'fruit']):
    strat_train_df = df.loc[tr_index]
    strat_valid_df = df.loc[val_index]
strat_train_df.shape,strat_valid_df.shape
strat_train_df.loc[:,'fruit'].value_counts()
img_width=300
img_height=300
train_data_IDG = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                    zoom_range=0.2, fill_mode='nearest', horizontal_flip=True, rescale=1.0/255)
valid_data_IDG = ImageDataGenerator(rescale=1.0/255, validation_split=0.05)
training_data = train_data_IDG.flow_from_dataframe(dataframe=strat_train_df,
                                                   y_col='fruit', x_col='path',
                                                   target_size=(img_width,img_height), batch_size=8, shuffle=True,
                                                   class_mode='categorical')
validation_data = valid_data_IDG.flow_from_dataframe(dataframe=strat_valid_df,
                                                     y_col='fruit', x_col='path',
                                                     target_size=(img_width,img_height), shuffle=True,
                                                     class_mode='categorical')
print(set(training_data.classes))
print(set(validation_data.classes))
tr_set_images, tr_set_labels = next(training_data)
val_set_images, val_set_labels = next(validation_data)
print(mapped_fruit_names[np.where(tr_set_labels[1] == 1)[0][0]])
plt.imshow(X=tr_set_images[1])
print(mapped_fruit_names[np.where(tr_set_labels[3] == 1)[0][0]])
plt.imshow(X=tr_set_images[3])
# Modelling

base_xception = Xception(include_top=False, weights='imagenet', input_shape=(299,299,3), pooling='max')
layer1 = base_xception.output
layer1 = layers.Dense(units=512, activation='relu')(layer1)
layer1 = layers.Dropout(rate=0.5)(layer1)
layer1 = layers.Dense(units=15, activation='softmax')(layer1)
model_1 = Model(inputs=base_xception.input, outputs=layer1)
model_1.summary()

# Lets compile our model :)
model_1.compile(loss = CategoricalCrossentropy(),
                optimizer = Adam(),
                metrics = [FBetaScore(num_classes=15, average='macro', name='fbeta_score'),
                           CategoricalAccuracy(name='cat_acc'),
                           Precision(name='precision'), Recall(name='recall')])
_early_stopping = EarlyStopping(monitor='val_fbeta_score', min_delta=0, patience=3, verbose=1, mode='max')
_reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, cooldown=2, min_lr=0.00001, verbose=1)
nm = pd.Series(training_data.classes).value_counts()
mapping = nm.rename("fruit").to_dict()
mapping
# scaling by 15
fruit_weights = {i:(1/mapping[i])*(sum(mapping.values()))/15.0 for i in range(15)}
fruit_weights
# Fitting model
train_model_1 = model_1.fit(training_data,validation_data=validation_data, class_weight=fruit_weights, verbose=1, epochs=17,
                      callbacks=[_early_stopping, _reduce_learning_rate])
model_1.save(filepath='/kaggle/working', include_optimizer=True, save_format='.tf')
plt.plot(history_model_1.history['loss'])
plt.plot(history_model_1.history['val_loss'])
plt.title('loss by epochs')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()
plt.plot(history_model_1.history['precision'])
plt.plot(history_model_1.history['val_precision'])
plt.plot(history_model_1.history['recall'])
plt.plot(history_model_1.history['val_recall'])
plt.title('Precision and Recall by epochs')
plt.ylabel('Precision and Recall')
plt.xlabel('epoch')
plt.legend(['train_precision', 'val_precision','train_recall', 'val_recall'], loc='best')
plt.show()