import dicom

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



print(check_output(["ls", "../input"]).decode("utf8"))



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)

print (os.system("ls sample_images"))

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)