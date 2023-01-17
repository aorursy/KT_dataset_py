# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image
import matplotlib.pyplot as plt
img=np.array(Image.open("/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0079038.jpg"))
#plt.imshow(img)
print(img.shape)
print(type(img))
df=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
df.head()
df['image_name'] = df['image_name'].apply(lambda x: f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg')
df.head()
X=np.array(Image.open(df['image_name'][0]))
X=np.resize(X,[32,32,3])
for i in range(1,len(df)):
    img=np.array(Image.open(df['image_name'][i]))
    img=np.resize(img,[32,32,3])
    X=np.dstack((X,img))
    print(i)
class config:
    image_width = 128
    image_height = 128
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3
    valid_size = 0.2
    base_model = 'se_resnext50_32x4d'
    seed = 0
    verbose_step = 1
# Split train and valid while making sure there is no patients simultaneously in train and valid
import random
unique_patient_ids = set(df['patient_id'])
unique_patient_ids = list(unique_patient_ids)
random.shuffle(unique_patient_ids)

train_ids = unique_patient_ids[:int( (1 - config.valid_size) * len(unique_patient_ids))]
valid_ids = unique_patient_ids[int( (1 - config.valid_size) * len(unique_patient_ids)):]

train_df = df[df['patient_id'].isin(train_ids)].sample(frac=1).reset_index(drop=True)
valid_df = df[df['patient_id'].isin(valid_ids)].sample(frac=1).reset_index(drop=True)

# Checking that there is no common patient id
a = set(train_df['patient_id'])
b = set(valid_df['patient_id'])
c = a.intersection(b)

assert len(c) == 0, 'Patients simultaneously in training and validation set'

# Checking the size
print(f'There are {len(train_df)} samples in the training set.')
print(f'There are {len(valid_df)} samples in the validation set.')
print(f'There are {len(train_df.query("target==1"))} in training set ({len(train_df.query("target==1")) / len(train_df) * 100: .2f} %)')
print(f'There are {len(valid_df.query("target==1"))} in validation set ({len(valid_df.query("target==1")) / len(valid_df) * 100: .2f} %)')
import tensorflow as tf
class MelanomaDataset:
    def __init__(self, image_paths, config, resize=True, augmentations=None):
        self.image_paths = image_paths
        #self.targets = targets
        self.augmentations = augmentations
        self.config = config
        self.resize = resize
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        #targets = self.targets[item]
        
        if self.resize:
            image = image.resize(
                (self.config.image_width, self.config.image_height), resample=Image.BILINEAR
            )
        
        image = np.array(image)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            'image': tf.tensor(image, dtype=tf.float),
            #'targets': tf.tensor(targets, dtype=tf.long),
        }
train_dataset = MelanomaDataset(
    image_paths=train_df['image_name'],
    #targets=train_df['target'],
    config=config,
    resize=True,
    augmentations=None,
)

#train_targets=tf.Tensor(train_df['target'], shape=(26316,), dtype=tf.float32)

valid_dataset = MelanomaDataset(
    image_paths=valid_df['image_name'],
    #targets=valid_df['target'],
    config=config,
    resize=True,
    augmentations=None,
 )

#test_targets=tf.Tensor(valid_df['target'],shape=(6810,), dtype=tf.float32)

print(train_df['image_name'].shape)
