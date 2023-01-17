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
train_V2=pd.read_csv('../input/train_V2.csv')

test_V2=pd.read_csv('../input/test_V2.csv')

submission=pd.read_csv('../input/sample_submission_V2.csv')
submission.head()
train_V2.winPlacePerc.describe()

train_V2.winPlacePerc.median()



















submission.head()

submission.winPlacePerc=0.4583

submission.head()
submission.to_csv('naive_submission.csv', index=False)
