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
import pandas as pd
import numpy as np
import os
from IPython.display import display
train_file = '..input/train.csv'
test_file = '..input/test.csv'
submission_file = 'submission.csv'
def PreparetrainData(in_file):
    full_data = pd.read_csv(in_file)
    display(full_data.head())
    outcomes = full_data['Survived']
    data = full_data.drop('Survived', axis = 1)
    return data, outcomes
data, outcomes = PreparetrainData(train_file)

