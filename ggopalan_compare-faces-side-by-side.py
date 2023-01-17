# Imports

import os



import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path
%matplotlib inline
# Some constants

HOME_DIR = Path('../input')

TRAIN_RELATION_CSV = HOME_DIR/'train_relationships.csv'

SAMPLE_SUBMISSION_CSV = HOME_DIR/'sample_submission.csv'

TRAIN_IMAGES = HOME_DIR/'train'

TEST_IMAGES = HOME_DIR/'test'
# Read the train relationships

train_relations_df = pd.read_csv(TRAIN_RELATION_CSV)
# Get paths to images from selected row

select_row = 20

image1_dir = TRAIN_IMAGES/train_relations_df.iloc[select_row].p1

image2_dir = TRAIN_IMAGES/train_relations_df.iloc[select_row].p2



# For now select the first image in each dir

image1_fns = [image1_dir/x for x in os.listdir(image1_dir)]

image2_fns = [image2_dir/x for x in os.listdir(image2_dir)]

# And then show them

min_num_images = min(len(image1_fns), len(image2_fns))

fig,ax = plt.subplots(min_num_images, 2, figsize=(10, min_num_images*5))



for i in range(min_num_images):

    # print(image1_fns[i])

    # print(image2_fns[i])



    image1 = plt.imread(image1_fns[i])

    image2 = plt.imread(image2_fns[i])

    if min_num_images == 1:

        ax[0].imshow(image1)

        ax[1].imshow(image2)

    else:

        ax[i][0].imshow(image1)

        ax[i][1].imshow(image2)



fig.show()