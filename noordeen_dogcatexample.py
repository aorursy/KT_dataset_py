import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout

from keras.layers import Dense, Flatten



import cv2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir('../input'))
!unzip /kaggle/input/train.zip -d /kaggle/working/
!unzip /kaggle/input/test.zip -d /kaggle/working/
category = ["cat", "dog"]



EPOCHS                  = 50

IMGSIZE                 = 128

BATCH_SIZE              = 32

STOPPING_PATIENCE       = 15

VERBOSE                 = 1

MODEL_NAME              = 'cnn_50epochs_imgsize128'

OPTIMIZER               = 'adam'

TRAINING_DIR            = '/kaggle/working/train'

TEST_DIR                = '/kaggle/working/test'
for img in os.listdir(TRAINING_DIR)[7890:]:

    img_path = os.path.join(TRAINING_DIR, img)

    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_arr = cv2.resize(img_arr, (IMGSIZE, IMGSIZE))

    plt.imshow(img_arr, cmap='gray')

    plt.title(img.split('.')[0])

    break
def create_train_data(path):

    X = []

    y = []

    for img in os.listdir(path):

        if img == os.listdir(path)[7889]:

            continue

        img_path = os.path.join(path, img)

        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_arr = cv2.resize(img_arr, (IMGSIZE, IMGSIZE))

        img_arr = img_arr / 255.0

        cat = np.where(img.split('.')[0] == 'dog', 1, 0)

        

        X.append(img_arr)

        y.append(cat)

    

    X = np.array(X).reshape(-1, IMGSIZE, IMGSIZE, 1)

    y = np.array(y)

    

    return X, y        
X, y = create_train_data(TRAINING_DIR)



print(f"features shape {X.shape}.\nlabel shape {y.shape}.")
y = to_categorical(y, 2)

print(f"features shape {X.shape}.\nlabel shape {y.shape}.")
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)