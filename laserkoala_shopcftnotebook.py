import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
train = pd.read_csv("../input/cft-shift-data/train.csv")

test = pd.read_csv("../input/cft-shift-data/test.csv")

users = pd.read_csv("../input/cft-shift-data/users.csv")

sample_submission = pd.read_csv("../input/cft-shift-data/sample_submission.csv")

users_items = pd.read_csv("../input/cft-users-and-items/users_items.csv")