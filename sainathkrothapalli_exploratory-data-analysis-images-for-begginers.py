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
data=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

data.head()
data.shape
data.isnull().sum()
data.info()
data['sex'].fillna(data['sex'].mode()[0],inplace=True)

data['sex']=data['sex'].map({'male': 1, 'female': 0})
data['age_approx'].fillna(data['age_approx'].median(),inplace=True)

data['anatom_site_general_challenge'].fillna(data['anatom_site_general_challenge'].mode()[0],inplace=True)
data.isnull().sum().sum()
data['target'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(data['target'])
plt.pie(data['target'].value_counts(),labels=[0,1],autopct='%1.2f%%')
sns.countplot(data['sex'])
plt.pie(data['sex'].value_counts(),labels=['male','female'],autopct='%1.2f%%')
sns.countplot(data['sex'],hue='target',data=data)
data.groupby(['sex'])['age_approx'].mean()
sns.kdeplot(data.loc[data['sex'] ==1, 'age_approx'], label = 'Male',shade=True)

sns.kdeplot(data.loc[data['sex'] ==0, 'age_approx'], label = 'Female',shade=True)
data.groupby(['target'])['age_approx'].mean()
sns.kdeplot(data.loc[data['target'] ==1, 'age_approx'], label = 'Malignant',shade=True)

sns.kdeplot(data.loc[data['target'] ==0, 'age_approx'], label = 'Bening',shade=True)
data['age_approx'].hist(color='g')
sns.violinplot(x='target',y='age_approx',data=data)
sns.barplot(x='sex',y='age_approx',hue='target',data=data)
data['anatom_site_general_challenge'].value_counts()
data['anatom_site_general_challenge'].value_counts().plot(kind='bar')
sns.countplot(x='anatom_site_general_challenge',hue='sex',data=data)

plt.xticks(rotation='vertical')
data['diagnosis'].value_counts()
IMAGE_PATH = "../input/siim-isic-melanoma-classification/"

from PIL import Image

images = data['image_name'].values

random_images = [np.random.choice(images+'.jpg') for i in range(9)]

img_dir = IMAGE_PATH+'/jpeg/train'

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = Image.open(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')
benign =data[data['benign_malignant']=='benign']

malignant =data[data['benign_malignant']=='malignant']

images = benign['image_name'].values

random_images = [np.random.choice(images+'.jpg') for i in range(9)]

img_dir = IMAGE_PATH+'/jpeg/train'

plt.figure(figsize=(10,8))

for i in range(9):

    plt.subplot(3, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')
images = malignant['image_name'].values

random_images = [np.random.choice(images+'.jpg') for i in range(6)]

img_dir = IMAGE_PATH+'/jpeg/train'

plt.figure(figsize=(10,8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')
Male =data[data['sex']==1]

Female =data[data['sex']==0]

images = Male['image_name'].values

random_images = [np.random.choice(images+'.jpg') for i in range(6)]

img_dir = IMAGE_PATH+'/jpeg/train'

plt.figure(figsize=(10,8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')
images = Female['image_name'].values

random_images = [np.random.choice(images+'.jpg') for i in range(6)]

img_dir = IMAGE_PATH+'/jpeg/train'

plt.figure(figsize=(10,8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')