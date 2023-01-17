# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
data.info()
print(f"Columns that we are using are :- {data.columns[0]}, {data.columns[1]}")
print(f"Total Numbers of the id or images {data.shape[0]}, with unique Landmarks {len(set(data['landmark_id']))}")
import matplotlib.pyplot as plt
landmarks = data['landmark_id']

ids = data['id']
#ids, landmarks
data['count'] = data["landmark_id"].value_counts()
data_count=data['count'].dropna()
xval = data_count.values
yval = np.array(list(set(landmarks)))
len(xval[:20]), len(yval[:20])
type(xval), type(yval)
import matplotlib.pyplot as plt



ax = plt.figure(figsize=(30, 10))

plt.hist(xval, bins='auto')

ax.show()
fig = plt.figure(figsize=(30, 10))

plt.plot(yval, xval, c = 'r', lw=2)

fig.show()
import os



images = ['../input/landmark-retrieval-2020/index/0/0/0/'+i for i in os.listdir('../input/landmark-retrieval-2020/index/0/0/0') if i.endswith('.jpg')]
images
import cv2

import matplotlib.pyplot as plt

import numpy as np



w=10

h=10

fig=plt.figure(figsize=(10, 8))

columns = 4

rows = 5

for i in range(1, len(images)):

    img = cv2.imread(images[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

plt.show()
lis = [f'../input/landmark-retrieval-2020/index/{l}/{j}/{k}/'+ i for l in range(2) for j in range(2) for k in range(1) for i in os.listdir(f'../input/landmark-retrieval-2020/index/{l}/{j}/{k}') if i.endswith(".jpg")]
len(lis)
import cv2

import matplotlib.pyplot as plt

import numpy as np



w=10

h=10

fig=plt.figure(figsize=(15, 15))

columns = 10

rows = 8

for i in range(1, len(lis)):

    img = cv2.imread(lis[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.title(lis[i])

plt.show()
import cv2

import matplotlib.pyplot as plt

import numpy as np



w=10

h=10

fig=plt.figure(figsize=(15, 15))

columns = 4

rows = 4

for i in range(1, len(lis[:9])):

    img = cv2.imread(lis[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.title(lis[i][-20:])

plt.show()