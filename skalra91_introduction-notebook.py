# import standard libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# load train data

train_data = pd.read_csv('/kaggle/input/syde522/train.csv')
# different classes

train_data.Category.unique()
# reading images

from skimage.io import imread

import matplotlib.pyplot as plt



%matplotlib inline



train_data['train_file'] = train_data.Id.apply(lambda x: '/kaggle/input/syde522/train/train/{0}'.format(x))





plt.figure(figsize=(15, 4))

for idx, (_, entry) in enumerate(train_data.sample(n=5).iterrows()):

    

    plt.subplot(1, 5, idx+1)

    plt.imshow(imread(entry.train_file))

    plt.axis('off')

    plt.title(entry.Category)
# preparing the submission



import glob

import os



def random_predictor(img):

    # this must be your glorious classifier and not a random predictor

    return train_data.sample(n=1).Category.iloc[0]



test_files = glob.glob('/kaggle/input/syde522/test/test/*.png')

test_file_id = [os.path.basename(test_file) for test_file in test_files]



test_submission = pd.DataFrame({'Id': test_file_id, 'Category': [random_predictor(test_img_id) for test_img_id in test_file_id]})



test_submission.to_csv('submission.csv', index=False)