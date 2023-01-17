import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import glob
def read_and_modify_dataframes(input_path):

    train_paths = glob.glob(input_path + 'train/*/*/*/*')

    test_paths = glob.glob(input_path + 'test/*/*/*/*')

    

    mapping = {}

    for path in train_paths:

        mapping[path.split('/')[-1].split('.')[0]] = path

    train_df = pd.read_csv(input_path + 'train.csv')

    train_df['image_path'] = train_df['id'].map(mapping)

    uniques = train_df['landmark_id'].unique()

    uniques_map = dict(zip(uniques, range(len(uniques))))

    train_df['label'] = train_df['landmark_id'].map(uniques_map).astype(np.int32)

    

    mapping = {}

    for path in test_paths:

        mapping[path.split('/')[-1].split('.')[0]] = path

    submission_df = pd.read_csv(input_path + 'sample_submission.csv')

    # remember to remove 'image_path' and 'label' when submitting:

    # ...

    # submission_df = submission_df.drop('label', axis=1)

    # submission_df = submission_df.drop('image_path', axis=1)

    # submission_df.to_csv('submission.csv')

    submission_df['image_path'] = submission_df['id'].map(mapping)

    submission_df['label'] = -1

    

    return train_df, submission_df



train_df, submission_df = read_and_modify_dataframes('../input/landmark-recognition-2020/')
train_df.head(20)
submission_df.head(20)
ID = 0

image_path = train_df.iloc[ID].image_path

label = train_df.iloc[ID].label

landmark_id = train_df.iloc[ID].landmark_id





image = Image.open(image_path)

plt.figure(figsize=(10, 10))

plt.imshow(image)

plt.title('label:' + str(label) + ', landmark_id:' + str(landmark_id),

          fontsize=20);