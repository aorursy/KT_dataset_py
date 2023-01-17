# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os import listdir
from os.path import isfile, join
input_dir = "../input"
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
data = {f.replace('.csv', ''): pd.read_csv(join(input_dir, f)) for f in files}
data.keys()
dev_id = data['gender_age_train'].sample()['device_id'].item()
dev_id
df = data['phone_brand_device_model']
df[df.device_id == dev_id]
df = data['events']
df2 = df[df.device_id == dev_id]
event_ids = df2['event_id'].tolist()
df2
data.keys()
