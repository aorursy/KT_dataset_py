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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.info())
print(test.info())
print(train.groupby('TARGET_5Yrs')['PlayerID'].count())
cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [1 for i in range(440)] }
submission = pd.DataFrame(cols)
print(submission)
submission.to_csv("submission.csv", index=False)