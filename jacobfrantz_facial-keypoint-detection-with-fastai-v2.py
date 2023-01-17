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
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from PIL import Image

from fastai.vision.all import *

from pathlib import Path

import pandas as pd

import tensorflow as tf
!pwd

!cd

!unzip -o /kaggle/input/facial-keypoints-detection/test.zip -d .

!unzip -o /kaggle/input/facial-keypoints-detection/training.zip -d .

!ls
# get files 



path = Path.cwd()

train = path/'training.csv'

test = path/'test.csv'



train_df = pd.read_csv(train, header='infer')

test_df = pd.read_csv(test)





#test_df.head() 

#train_df.describe() 

train_df.columns
# how to get an image from a row of the dataset

def str2img(row):

  imarr = np.fromstring(row.Image, dtype='int32', sep=' ').astype(np.int32)

  i = Image.fromarray(imarr.reshape(-1, 96)).convert('P')

  return PILImage(i)



# how to get the keypoints from a row of the dataset

def row2points(r): 

  a = np.reshape(r[0:30].values, (15,2)).astype(np.float64)

  return a



#type(PILImage(str2img(train_df.iloc[5298])))
# investigate which columns are null

label_df = train_df[train_df.columns[:-1]] # -1 bc last row is img

nulls_by_row = label_df.isnull().sum(axis=1)

nulls_by_row.plot()

nulls_by_row.value_counts() # most common nulls are 22 (4755 rows) and 0 (2140 rows)
##  deal with missing values



## Option A: only take "full" rows

#label_df = train_df[train_df.columns[:-1]] #

#train_df = train_df.loc[label_df.notnull().sum(axis=1) == 30]



## Option B: fill everything with prev value

train_df = train_df.fillna(method='ffill')



#train_df[train_df.columns[:-1]].describe()

#train_df[train_df.columns[:-1]] = train_df[train_df.columns[:-3]].fillna(train_df[train_df.columns[:-3]].mean())
db = DataBlock(

    blocks = (ImageBlock, PointBlock),

    get_x = str2img,

    get_y = row2points,

    splitter = RandomSplitter(valid_pct=0.15, seed=42),

    batch_tfms = aug_transforms(do_flip=False, max_zoom=1.0), # should prob adjust these params    

)

dls = db.dataloaders(train_df)

dls.show_batch()
dls.train.show_batch()

dls.valid.show_batch()
learn = cnn_learner(dls, resnet152)

learn.lr_find()
learn.fine_tune(10) # should really do like 50

learn.show_results()

learn.save('after-first-finetune')
# function to sanity check and play around with results

def show_pred(i, df):

    pic = PILImage(str2img(df.iloc[i]))

    (pred, t1, t2) = learn.predict(pic) #don't know what t1 and t2 are

    pic = TensorImage(pic)

    screen = pic.show()

    pred.show(ctx=screen)

    

show_pred(1000, test_df)
preds = [learn.predict(PILImage(str2img(test_df.iloc[idx])))[0] for idx in range(len(test_df))]

results = [x.reshape(30).numpy() for x in preds]

results = pd.DataFrame(results)

results.head()
results.shape



names = [

         'left_eye_center_x','left_eye_center_y',

         'right_eye_center_x','right_eye_center_y',

         'left_eye_inner_corner_x','left_eye_inner_corner_y',

         'left_eye_outer_corner_x','left_eye_outer_corner_y',

         'right_eye_inner_corner_x','right_eye_inner_corner_y',

         'right_eye_outer_corner_x','right_eye_outer_corner_y',

         'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',

         'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',

         'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',

         'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',

         'nose_tip_x','nose_tip_y',

         'mouth_left_corner_x','mouth_left_corner_y',

         'mouth_right_corner_x','mouth_right_corner_y',

         'mouth_center_top_lip_x','mouth_center_top_lip_y',

         'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y'

]

dicty = {}

for x in range(30):

  dicty[x] = names[x]

print(dicty)

results.rename(dicty, axis='columns', inplace=True)

results['ImageId'] = range(1, 1783+1)

#results.head()
# need to go from point values as COLUMNS to point values as ROWS

sorted_results = results.melt(id_vars='ImageId', value_vars=names).sort_values(by=['ImageId'])

sorted_results['variable'] = pd.CategoricalIndex(sorted_results['variable'], names)

sorted_results.rename(columns={'variable':'FeatureName', 'value':'Location'}, inplace=True)

sorted_results.sort_values(by=['ImageId','FeatureName'], inplace=True)

sorted_results.set_index(['ImageId','FeatureName'], inplace=True)

#sortrez.head(30)



# and then put predictions the way the submission file wants it



look = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

look.set_index(['ImageId','FeatureName'], inplace=True)

look.head()



combo = look.join(sorted_results, on=['ImageId','FeatureName'], lsuffix='remove')

combo.drop(columns='Locationremove', inplace=True)

combo.reset_index(inplace=True)

combo[['RowId','ImageId','FeatureName','Location']]

combo['Location'] = combo['Location'].clip(lower=0, upper=96)

combo.describe()

#combo
combo[['RowId','Location']].to_csv('submission.csv', index=False)

# ran with like 70 epochs and got ~top third of leaderboard