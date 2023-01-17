import fastai

from fastai.vision import *

import shutil

import os

import pandas as pd

work_dir = Path('/kaggle/working/')

path = Path('../input/widsdatathon2019')



source=path/'leaderboard_test_data'

destination= Path('../input/psuedo_images/')



train = path/'train_images/train_images'

test =  path/'leaderboard_test_data/leaderboard_test_data'

holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'

sample_sub = path/'SampleSubmission.csv'

labels = path/'traininglabels.csv'

test_labels = '../input/sub-test/test_label.csv'

holdout
! ls ../input/widsdatathon2019
sub099912 =pd.read_csv('../input/subs-wids/sub_099912.csv')

sub099911 =pd.read_csv('../input/subs-wids/11_folds_median.csv/11_folds_median.csv')
import os

def get_test(img_id) :

    

    if os.path.isfile(test/img_id): 

        return 'test'

    if os.path.isfile(holdout/img_id): 

        return 'holdout'     

    return ''   

    
sub099912['source'] = sub099912['image_id'].apply(lambda x: get_test(x) )

sub099912.head()
sub099911['source'] = sub099911['image_id'].apply(lambda x: get_test(x) )

sub099912.head()
sub099912.groupby(['source']).size()
sub099911.groupby(['source']).size()
sub099912.groupby(['source'])['has_oilpalm'].mean()
sub099911.groupby(['source'])['has_oilpalm'].mean()
sub099912=sub099912.sort_values(by=['has_oilpalm'],ascending=False)

sub099912.head()
sub099912_holdout = sub099912.loc[sub099912['source']=='holdout'].reset_index()

sub099912_holdout.head(10)
import cv2

def readImage(path):

    # OpenCV reads the image in bgr format by default

    bgr_img = cv2.imread(path)

    # We flip it to rgb for visualization purposes

    b,g,r = cv2.split(bgr_img)

    rgb_img = cv2.merge([r,g,b])

    return rgb_img
for k in range(25) :

    

    long=20

    larg=5

    sz = long*larg

    fig, ax = plt.subplots(long,larg, figsize=(20,5*long))

    fig.suptitle(str(k*sz) + ' - ' + str((k*sz)+sz),fontsize=20)



    for i, idx in enumerate(sub099912_holdout['image_id'][k*sz:(k*sz)+sz]):

        path = os.path.join('../input/widsdatathon2019/leaderboard_holdout_data/leaderboard_holdout_data/', idx)

        j = i // larg

        ax[j,i-(larg*j)].imshow(readImage(path))

        ax[j,i-(larg*j)].set_ylabel(sub099912_holdout['has_oilpalm'].iloc[i+k*sz], size='large',color='red')

    
