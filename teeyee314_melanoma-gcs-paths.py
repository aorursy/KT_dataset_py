import numpy as np

import pandas as pd

from glob import glob

import os

from kaggle_datasets import KaggleDatasets
gcs_paths = [x.split('/')[2:][0] for x in glob('../input/*')]

gcs_paths.remove('siim-isic-melanoma-classification')

gcs_paths.sort()

gcs_paths
path_dict = {}



for path in gcs_paths:

    path_dict[path] = KaggleDatasets().get_gcs_path(path)

    print(f'{path}\t| {KaggleDatasets().get_gcs_path(path)}')
# copy and paste the dictionary into colab

path_dict
# modify the GCS_PATHs on colab



# GCS_PATH[i] = path_dict[f'melanoma-{k}x{k}']

# GCS_PATH2[i] = path_dict[f'isic2019-{k}x{k}']