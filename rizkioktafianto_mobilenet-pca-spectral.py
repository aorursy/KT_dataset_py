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

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import cv2
from PIL import Image

from keras import layers
from tensorflow.keras import applications 
from keras.applications import MobileNetV2
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from tqdm import tqdm
df = pd.read_csv('../input/keep-babies-safe/dataset/test.csv')
print(df.shape)
df.head()
def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'Image']
        img = cv2.imread(f'../input/keep-babies-safe/dataset/images/{image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(df)
def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.BILINEAR)
    
    return im

N = df.shape[0]
x= np.empty((N, 224, 224, 3), dtype=np.float32)

for i, image_path in enumerate(tqdm(df['Image'])):
    x[i, :, :, :] = preprocess_image(
        f'../input/keep-babies-safe/dataset/images/{image_path}'
    )
mnet_model = MobileNetV2(input_shape=(224,224,3),
                        weights="imagenet",
                        include_top=False)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data = ImageDataGenerator()

def build_model():
    model = Sequential()
    model.add(mnet_model)
    model.add(layers.GlobalAveragePooling2D())
    
    return model

model = build_model()
model.summary()

train_features = model.predict(x)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans, SpectralClustering

pipe = Pipeline(steps=[("PCA", KernelPCA(n_components=12, kernel="rbf")),
                       ("cluster", SpectralClustering(n_clusters=2))])

y = pipe.fit_predict(train_features)
def display_samples(df, columns=4, rows=10):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'Image']
        img = cv2.imread(f'../input/keep-babies-safe/dataset/images/{image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(y[i])
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(df)