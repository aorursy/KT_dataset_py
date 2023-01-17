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


import pandas as pd

import os, sys

import glob

import cv2

from keras.utils import to_categorical

import keras

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import cv2                  

import numpy as np  

from tqdm import tqdm, tqdm_notebook

import os, random

from random import shuffle  

from zipfile import ZipFile

from PIL import Image

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

import fastai

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

from fastai.basic_train import *

from fastai.vision.learner import *

def is_interactive():

   return 'runtime' in get_ipython().config.IPKernelApp.connection_file



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(42)



Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

PATH = Path('../input/aptos2019-blindness-detection')



df_train = pd.read_csv(PATH/'train.csv')

df_test = pd.read_csv(PATH/'test.csv')





aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])

data = ImageDataBunch.from_df(df=df_train,

                              path=PATH, folder='train_images', suffix='.png',

                              valid_pct=0.1,

                              ds_tfms=get_transforms(flip_vert=True, max_warp=0.1, max_zoom=1.15, max_rotate=45.),

                              size=224,

                              bs=32, 

                              num_workers=os.cpu_count()

                             ).normalize(aptos19_stats)

data.show_batch(rows=3, figsize=(7,6))

os.listdir('../input/')

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_list = [[PIL.Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png'),j] for i,j in zip(train.id_code[:5],train.diagnosis[:5])]

train_list



for i,j in train_list:

    plt.figure(figsize=(5,3))

    i = cv2.resize(np.asarray(i),(256,256))

    plt.title(j)

    plt.imshow(i)

    plt.show
x_train = [cv2.resize(np.asarray(PIL.Image.open('../input/aptos2019-blindness-detection/train_images/'+i+'.png')),(256,256)) for i in train.id_code]
x_train = np.array(x_train)

y_train = train.diagnosis

y_train = to_categorical(y_train)
model = keras.applications.densenet.DenseNet121(input_shape=(256,256,3),include_top=True,weights=None)



model.summary()
x = model.layers[-2].output

d = keras.layers.Dense(512,activation='relu')(x)

e = keras.layers.Dense(5,activation='softmax')(d)

model1 = keras.models.Model(model.input,e)

model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model1.fit(x_train,y_train,validation_split=0.20,epochs=10)
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test = []

for i in test_df.id_code:

    temp = np.array(cv2.resize(np.array(PIL.Image.open('../input/aptos2019-blindness-detection/test_images/'+i+'.png')),(256,256)))

    test.append(temp)

test = np.array(test)



np.random.seed(42)

result = model1.predict(test)

res = []

for i in result:

    res.append(np.argmax(i))

df_test = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": res})

df_test.head(30)






