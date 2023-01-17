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
df_labels = pd.read_csv('../input/train_human_labels.csv')
df_labels.sample(2)
label_counts = df_labels['LabelName'].value_counts()
predict = ' '.join(label_counts[:6].index)
predict
df_submission = pd.read_csv('../input/stage_1_sample_submission.csv')
df_submission.head(2)
df_submission['labels'] = predict
df_submission.to_csv('submission.csv', index=False)
