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
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
test.head()
print("Train data shape : rows :-{0} , columns:- {1}".format(train.shape[0], train.shape[1]))
train.info()
print('Number of empty records :', train.isnull().any().sum())

path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'

for i in train['image_id'][:5]:
    print(i)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
image_index = 5
full_path = path + train['image_id'][image_index] + '.jpg'
img = Image.open(full_path)
plt.imshow(img)

print('Image default size', img.size)

train_healthy = train[train['healthy'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('Healthy')
plt.show()
train_healthy = train[train['scab'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('scab')
# plt.show()
train_healthy = train[train['rust'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('rust')
plt.show()
train_healthy = train[train['multiple_diseases'] == 1][:5]

fig, axs = plt.subplots(1,5, figsize=(25,6))

for i, j in enumerate(train_healthy['image_id']):
    axs[i].set_axis_off()
    print(path + j + '.jpg')
    full_path = path + j + '.jpg'
    img = Image.open(full_path)
    axs[i].imshow(img)
    axs[i].set_title('multiple_diseases')
plt.show()
train.shape
train.head()
data = train[['healthy', 'multiple_diseases', 'rust', 'scab']].sum(axis=0)
data.plot(kind='bar')
plt.title('Frequency count')

## pie chart
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

data.plot(kind='pie', colors=colors, title='Data with pie chart', figsize=(10,10)
)

from keras.preprocessing import image 
import numpy as np
img_size  = 224 # image size during training 
load_features = []

for i in train['image_id']:
    full_path = path + i + '.jpg'
    img = image.load_img(full_path, target_size=(img_size,img_size,3), color_mode = "rgb")
    img = image.img_to_array(img)
    load_features.append(img)
X = np.asarray(load_features)
print('Shape of features:',X.shape)
train.head()
y = train.iloc[:, 1:]
y = y.to_numpy()

y.shape
y[:5]
y.sum(axis=0)
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=42)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)
x_train_final = x_train /255
x_train_final.shape
x_test_final = x_test /255
x_test_final.shape
y_train_final = y_train
y_train_final.shape
y_test_final = y_test
y_test_final.shape

