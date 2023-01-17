# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/wec_24/train.csv')
train['Label'].describe()
group = train.groupby('Label').agg('count')['duration']
group = group/train.shape[0]
sub = pd.read_csv('../input/wec_24/SampleSubmission.csv')
for category in group.index:
    sub.loc[:, category] = group.loc[category]
sub.to_csv('statistical_sub.csv', index = False)
