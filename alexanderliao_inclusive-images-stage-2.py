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
import pandas as pd #just compress Anton Petrov's code in 6 lines
d1 = pd.read_csv('../input/inclusive-images-challenge/tuning_labels.csv', names=['id', 'labels'], index_col=['id'])
d2 = pd.read_csv('../input/inclusive-images-state-2/stage_2_sample_submission.csv', index_col='image_id')
d2['labels'] = ' '.join(d1['labels'].str.split().apply(pd.Series).stack().value_counts().head(3).index.tolist())
d2.update(d1)
d2.to_csv('last_day_6_lines_baseline.csv')
