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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
DATASET_DIR = "../input/covid-detection-project/"
metadata_path = str(DATASET_DIR) + 'metadata.csv'
df = pd.read_csv(metadata_path)
df.shape
df.head()
# These 2 images are not in the directory
df.drop(df.loc[(df['Filename'].isin(['COVID-19(34).png', 'COVID-19(84).png']))].index, inplace=True)
balanced_male_covid = df.loc[(df['Disease'] == 'COVID-19') & (df['Gender'] == 'M')].head(77)
balanced_female_covid = df.loc[(df['Disease'] == 'COVID-19') & (df['Gender'] == 'F')].head(77)



balanced_male_normal = df.loc[(df['Disease'] == 'normal') & (df['Gender'] == 'M')].head(77)
balanced_female_normal = df.loc[(df['Disease'] == 'normal') & (df['Gender'] == 'F')].head(77)

balanced_male_pneumonia = df.loc[(df['Disease'] == 'pneumonia') & (df['Gender'] == 'M')].head(77)
balanced_female_pneumonia = df.loc[(df['Disease'] == 'pneumonia') & (df['Gender'] == 'F')].head(77)

Covid = pd.concat([balanced_male_covid,balanced_female_covid])
Normal = pd.concat([balanced_male_normal,balanced_female_normal])
Pneumonia = pd.concat([balanced_male_pneumonia,balanced_female_pneumonia])
Covid.shape
dataframe_balanced['Gender'].value_counts()
cnt = 0
images = []
labels = []
IMAGES_PATH = DATASET_DIR +  'Covid/'
for(i,row) in Covid.iterrows():
    filename = row["Filename"]
    image_path = os.path.join(IMAGES_PATH,filename)
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))



    images.append(image)
    labels.append('Covid')
    cnt = cnt + 1

print(cnt)
IMAGES_PATH = DATASET_DIR +  'Normal/'
for(i,row) in Normal.iterrows():
    filename = row["Filename"]
    image_path = os.path.join(IMAGES_PATH,filename)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    images.append(image)
    labels.append('Normal')
    cnt = cnt + 1
print(cnt)


IMAGES_PATH = DATASET_DIR +  'Pneumonia/'
for(i,row) in Pneumonia.iterrows():
    filename = row["Filename"]
    image_path = os.path.join(IMAGES_PATH,filename)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    images.append(image)
    labels.append('Pneumonia')
    cnt = cnt + 1
print(cnt)
fig = plt.figure()
fig.suptitle('covid')
plt.imshow(images[0], cmap='gray') 
