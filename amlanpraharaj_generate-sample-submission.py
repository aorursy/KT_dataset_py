# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/wec_24"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/wec_24/train.csv')
train_data.head()
test_data = pd.read_csv('../input/wec_24/train.csv')
sub = test_data['Id']
sub = pd.concat((sub, pd.get_dummies(train_data['Label'].reset_index(drop=True))), axis = 1)
sub.loc[:, 1:] = 0.5
sub.to_csv('SampleSubmission.csv', index=False)
