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


import numpy as np
import pandas as pd
import pandas.util.testing as tm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from ast import literal_eval 
import datetime

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option("display.max_columns", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 50)
train_csv=pd.read_csv("../input/super-ai-image-classification/train/train/train.csv")
train_csv.head()
print(train_csv.shape)
train_csv.groupby("category").count().plot.bar()
import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as style

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from keras.preprocessing import image
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tensorflow.keras import layers

#from utils import *

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
print("TF version:", tf.__version__)
X_train_csv=train_csv['id'].tolist()
Y_train_csv=np.array(train_csv['category'])
train_csv['str_category']= train_csv['category'].apply(lambda x: str(x))
X_train_img = [os.path.join("../input/super-ai-image-classification/train/train/images/", str(i)) for i in X_train_csv]
X_train_img[:3]

nobs = 12 # Maximum number of images to display
ncols = 4 # Number of columns in display
nrows = nobs//ncols # Number of rows in display

style.use("default")
plt.figure(figsize=(12,4*nrows))
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    plt.imshow(Image.open(X_train_img[i+1000]))
    plt.title(Y_train_csv[i+1000], size=10)
    plt.axis('off')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
dataframe=train_csv,
directory="../input/super-ai-image-classification/train/train/images",
x_col="id",
y_col="str_category",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(224,224))


validation_generator=datagen.flow_from_dataframe(
dataframe=train_csv,
directory="../input/super-ai-image-classification/train/train/images",
x_col="id",
y_col="str_category",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="binary",
target_size=(224,224))
from keras.applications import InceptionResNetV2
from tensorflow.keras import layers
import keras
import numpy as np
from keras.applications import InceptionResNetV2
from tensorflow.keras import layers
#Load the VGG19 model


ResNetV2 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))


# fit input
x_ResNetV2 = layers.Flatten()(ResNetV2.output)
x_ResNetV2 = layers.Dense(1, activation='sigmoid')(x_ResNetV2)
model_ResNetV2 = keras.Model(ResNetV2.input, x_ResNetV2)
model_ResNetV2.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model_ResNetV2.compile(loss='binary_crossentropy',optimizer= keras.optimizers.SGD(learning_rate=0.0001),metrics=['accuracy'])
from time import time
start = time()
history = model_ResNetV2.fit(train_generator,
                    epochs=150,
                  validation_data=validation_generator,callbacks=[callback])
# load all images into a list
folder_path='../input/super-ai-image-classification/val/val/images'
images = []
yhat=[]
for i in os.listdir(folder_path):
    img = os.path.join(folder_path, i)
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)/255.
    img = np.expand_dims(img, axis=0)
    y=model_ResNetV2.predict(img)
    y_round =1 if y[0][0] >= 0.5 else 0
    yhat.append([i,y_round])
df=pd.DataFrame(yhat)
df.rename(columns={0:'id',1:'category'},inplace=True)
df.to_csv('22p22w0030_t3.csv',index=False)