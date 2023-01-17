## Notebook Initiation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # loading files
# Load files into 1 dataframe

kaggle_path = '/kaggle/input/'

donation_files = pd.DataFrame()

for dirname, _, filenames in os.walk(kaggle_path):

    for file in filenames:

        print(dirname+'/'+file)

        donation_files = donation_files.append(pd.read_csv(dirname+'/'+file))