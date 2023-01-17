# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
sample_sub = pd.read_csv("../input/sample_submission_V2.csv")
# Any results you write to the current directory are saved as output.
train.head()
train_nums = train.drop(columns=['matchId','groupId','Id','matchType'])
train_nums = train[['kills','walkDistance','rideDistance','killStreaks','winPlacePerc']]
sns.heatmap(train_nums.corr())
