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
import glob

#csv_list = sorted(glob.glob('../input/o2unet/*.csv'))

csv_list = sorted(glob.glob('../input/o2unet2/*.csv'))

pd.get_option("display.max_columns")

pd.set_option('display.max_columns', 200)

pd.get_option("display.max_rows")

pd.set_option('display.max_rows', 200)
def get_normalized_loss(csv_name):

    df = pd.read_csv(csv_name)

    df = df.loc[:,['img_idx','loss']].set_index('img_idx')

    print(df['loss'].mean())

    #print(os.path.basename(csv_name).split('_')[0])

    df['loss'] = df['loss'] - df['loss'].mean()

    return df.sort_index().add_prefix(os.path.basename(csv_name).split('_')[0] + '_')

get_normalized_loss(csv_list[0])
df = get_normalized_loss(csv_list[0])

for csv_name in csv_list[1:]:

    df = df.join(get_normalized_loss(csv_name)) 

df[:200]
#df['loss_avg'] = df.var(axis='columns')

df['loss_avg'] = df.mean(axis='columns')

df = df.sort_values('loss_avg')

df[:200]
k=0.1

num_remain = int(len(df)*(1-k))

print('remain: ' + str(num_remain))

print('')

print('cutting line:')

print(df.iloc[num_remain])

df['loss_avg'].hist()
#df[df['loss_avg']>1.974052]['loss_avg'].hist()

df[df['loss_avg']>1.974052]['loss_avg'].hist()

print(len(df[df['loss_avg']>2**2]))
train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

remain_df = df[:num_remain]

remain_df.index.values
train_df_clean = train_df.loc[remain_df.index.values]
train_df_clean['data_provider'].value_counts().plot(kind="bar")
train_df_clean[:200]
train_df_clean[-200:]
train_df_clean['isup_grade'].hist()
delete_df = df[num_remain:]

train_df_delete = train_df.loc[delete_df.index.values]

train_df_delete['isup_grade'].hist()
train_df_delete[-200:]
train_df_delete['data_provider'].value_counts().plot(kind="bar")

#train_df_clean['data_provider'][:1000].value_counts().plot(kind="bar")
train_df_delete.to_csv('o2u-22017707-k01.csv')
df.loc[5880]
!pip install imagecodecs

!pip install tifffile
!dpkg -l | grep pixman
import skimage.io

import matplotlib.pyplot as plt



idx = 0
plt.figure(figsize=(6,6))

img_id = train_df_delete.iloc[-idx]['image_id']

img_path = '../input/prostate-cancer-grade-assessment/train_images/' + img_id + '.tiff'



image = skimage.io.MultiImage(img_path)[2]

image = np.array(image)

plt.imshow(image)

idx += 1