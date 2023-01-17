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
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
from tqdm import tqdm, tqdm_notebook
import gc
import seaborn as sns
sns.set_style("dark")
#Image data
import os
image_train_dir = '../input/siim-isic-melanoma-classification/jpeg/train'
listtrain = os.listdir(image_train_dir) # dir is your directory path
number_files_train = len(listtrain)
print('Number of images in Train dataset:',number_files_train)

image_test_dir = '../input/siim-isic-melanoma-classification/jpeg/test'
listtest = os.listdir(image_test_dir) # dir is your directory path
number_files_test = len(listtest)
print('Number of images in Test dataset:',number_files_test)
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
img_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')

fig, ax = plt.subplots(4, 4, figsize=(20, 20))

for i in range(16):
    x = i // 4
    y = i % 4
    
    path = img_names[i]
    image_id = path.split("/")[5][:-4]
    
    target = train_df.loc[train_df['image_name'] == image_id, 'target'].tolist()[0]
    
    img = Image.open(path)
    
    ax[x, y].imshow(img)
    ax[x, y].axis('off')
    ax[x, y].set_title(f'ID: {image_id}, Target: {target}')

fig.suptitle("Training set samples", fontsize=15)
path_train = '../input/siim-isic-melanoma-classification/jpeg/train/'


train_df_malignent = train_df[train_df['benign_malignant'] == 'malignant']
train_df_benign = train_df[train_df['benign_malignant'] == 'benign']

fig, ax = plt.subplots(4, 4, figsize=(20, 20))

for i in range(16):
    x = i // 4
    y = i % 4
    image_id = train_df_benign.iloc[i,0]
    path = path_train + image_id + '.jpg'
    image_id = path.split("/")[5][:-4]
    
    
    img = Image.open(path)
    
    ax[x, y].imshow(img)
    ax[x, y].axis('off')
    ax[x, y].set_title(f'ID: {image_id}')

fig.suptitle("Benign samples", fontsize=15)
fig, ax = plt.subplots(4, 4, figsize=(20, 20))

for i in range(16):
    x = i // 4
    y = i % 4
    image_id = train_df_malignent.iloc[i,0]
    path = path_train + image_id + '.jpg'
    image_id = path.split("/")[5][:-4]
    
    
    img = Image.open(path)
    
    ax[x, y].imshow(img)
    ax[x, y].axis('off')
    ax[x, y].set_title(f'ID: {image_id}')

fig.suptitle("Malignent samples", fontsize=15)

image_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')
size_array_train = []

for image_name in tqdm(image_names):
    path = image_name
    img = Image.open(path)
    temp = img.size
    size_array_train.append(temp)
    
len(size_array_train)
image_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/test/*.jpg')
size_array_test = []

for image_name in tqdm(image_names):
    path = image_name
    img = Image.open(path)
    temp = img.size
    size_array_test.append(temp)
    
len(size_array_test)
size_df_train = pd.DataFrame(np.row_stack(size_array_train))
size_df_train.describe()
size_df_test = pd.DataFrame(np.row_stack(size_array_test))
size_df_test.describe()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

g1 = sns.distplot(size_df_train[0], ax=ax[0])
ax[0].set_title("training set")

g2 = sns.distplot(size_df_test[0], ax=ax[1])
ax[1].set_title("test set")

# g1.set_xticklabels(g1.get_xticklabels(), rotation=45)
# g2.set_xticklabels(g2.get_xticklabels(), rotation=45)
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

g1 = sns.distplot(size_df_train[1], ax=ax[0])
ax[0].set_title("training set")

g2 = sns.distplot(size_df_test[1], ax=ax[1])
ax[1].set_title("test set")

# g1.set_xticklabels(g1.get_xticklabels(), rotation=45)
# g2.set_xticklabels(g2.get_xticklabels(), rotation=45)
plt.show()
train_df_malignent.head(5)
image_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')
size_array_train = []
i = 0

for image_name in tqdm(image_names):
    temp = image_name.split('/')
    temp1 = temp[-1]
    temp2 = temp1.split('.')
#     print(temp2[0])
    if(train_df_benign['image_name'].str.contains(temp2[0]).any()):
        print(temp2)
    print(temp2[0])
    i = i + 1
    if(i == 4):
        break
image_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')
size_array_train = []
i = 0

for image_name in tqdm(image_names):
    temp = image_name.split('/')
    temp1 = temp[-1]
    temp2 = temp1.split('.')
#     print(temp2[0])
    if(train_df_benign['image_name'].str.contains(temp2[0]).any()):
        print(temp2)
    print(temp2[0])
    i = i + 1
    if(i == 4):
        break

image_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')
size_array_train_benign = []

for image_name in tqdm(image_names):
    path = image_name
    temp = image_name.split('/')
    temp1 = temp[-1]
    temp2 = temp1.split('.')
    if(train_df_benign['image_name'].str.contains(temp2[0]).any()):
        img = Image.open(path)
        temp = img.size
        size_array_train_benign.append(temp)
    
len(size_array_train_benign)

image_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')
size_array_train_malignent = []

for image_name in tqdm(image_names):
    path = image_name
    temp = image_name.split('/')
    temp1 = temp[-1]
    temp2 = temp1.split('.')
    if(train_df_malignent['image_name'].str.contains(temp2[0]).any()):
        img = Image.open(path)
        temp = img.size
        size_array_train_malignent.append(temp)
    
len(size_array_train_malignent)