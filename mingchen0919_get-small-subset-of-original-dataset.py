import os

import random

from shutil import copy, make_archive





data_root = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'

k = 100 # randomly select 100 images from both train and test data folder

os.makedirs('./dataset', exist_ok=True) # create a dataset folder to hold all the files that I wanted to download

copy(os.path.join(data_root, 'stage_2_sample_submission.csv'), 'dataset/stage_2_sample_submission.csv')

copy(os.path.join(data_root, 'stage_2_train.csv'), 'dataset/stage_2_train.csv')

for d in ['stage_2_train', 'stage_2_test']:

    # list all images in train/test folder

    dir_path = os.path.join(data_root, d)

    files = os.listdir(dir_path)

    

    # copy images to target folder

    target_dir = os.path.join('dataset', d)

    os.makedirs(target_dir, exist_ok=True) 

    for f in random.choices(files, k=k): # randomly select k images and copy them to the target folder

        src_file = os.path.join(dir_path, f)

        copy(src_file, target_dir)

        

# zip generated files

make_archive(base_name='download_dataset', format='zip', root_dir='dataset')
