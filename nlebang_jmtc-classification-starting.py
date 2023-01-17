# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DIR_INPUT= '/kaggle/input/jigsaw-multilingual-toxic-comment-classification'

train_df1 = pd.read_csv(DIR_INPUT + '/jigsaw-toxic-comment-train.csv')

train_df1.head()
print(train_df1.shape)
train_df2 = pd.read_csv(DIR_INPUT + '/jigsaw-unintended-bias-train.csv')

train_df2.head()
print(train_df2.shape)

print(train_df2.columns)
validation_df = pd.read_csv(DIR_INPUT + '/validation.csv')

validation_df.head()
print(validation_df.shape)
test_df = pd.read_csv(DIR_INPUT + '/test.csv')

test_df.head()
print(test_df.shape)
sample_submission= pd.read_csv(DIR_INPUT + '/sample_submission.csv')

sample_submission.head()
print(sample_submission.shape)