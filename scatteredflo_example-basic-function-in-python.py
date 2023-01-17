import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv("../input/dlp-private-competition-dataset-modificated/train.csv", index_col = ["ID"])

train.head()