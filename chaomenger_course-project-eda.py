# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm

from tqdm import tqdm, tqdm_notebook

import cv2

from PIL import Image
num_classes = 11

batch_size = 32

img_size = 224

num_epochs = 15



flower_name = {"0":"dahlia", 

               "1":"daisy",

               "2":"Dandelion",

               "3":"Daylily",

               "4":"Evergreen",

               "5":"iris",

               "6":"narcissus",

               "7":"peony",

               "8":"rose",

               "9":"sunflower",

               "10":"tulip"}
data = pd.read_csv(r"/kaggle/input/course-project-flower/data.csv")

data.head()
for i in range(data.shape[0]):

    p = "/kaggle/input/course-project-flower/pic/pic/" + data.iloc[i][0][22:]

    data.loc[i, "path"] = p

data.head()
N = data.shape[0]

X = np.empty((N, img_size, img_size, 3), dtype=np.uint8)

Y = np.empty(N, dtype=np.uint8)

arr = np.arange(3796)

np.random.shuffle(arr)



for i in tqdm_notebook(arr):

    p = data.iloc[i][0]

    img = cv2.imread(p)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_size,img_size))

    X[i, :, :, :] = img

    Y[i] = data.iloc[i][1]

print(N)
print(X.shape)

print(Y.shape)
import random

figure, ax = plt.subplots(1, 2, figsize=(10, 5))

cnt = random.randint(0,3796)

p = data.iloc[cnt][0]

print(cnt, p)

img = cv2.imread(p)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

ax[0].imshow(img)

ax[0].set_title("origin_"+flower_name[str(data.iloc[cnt][1])])

img = cv2.resize(img, (img_size,img_size))

ax[1].imshow(img)

ax[1].set_title("resize_"+flower_name[str(data.iloc[cnt][1])])

        

plt.show()
for i in range(data.shape[0]):

    data.loc[i, "flower"] = str(data.iloc[i][1]) + "_" + flower_name[str(data.iloc[i][1])]

data.head()
data.shape
plt.figure(figsize=(12, 6))

sns.countplot(data["flower"])

plt.title("Number of flower per each class")

plt.show()