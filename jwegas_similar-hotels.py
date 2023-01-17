# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
test_one = pd.read_csv('../input/test.csv', nrows=1)
train_one = pd.read_csv('../input/train.csv', nrows=1)
test_one
train_chunk = pd.read_csv('../input/train.csv', chunksize=10000)
train = pd.DataFrame(columns=train_one.keys())

for chunk in train_chunk:
    train = pd.concat([train, chunk[  (chunk['hotel_country'] == test_one['hotel_country'].values[0]) \
                                   & (chunk['srch_destination_id'] == test_one['srch_destination_id'].values[0])\
                                   & ((chunk['srch_children_cnt'] > 0) == (test_one['srch_children_cnt'].values[0] > 0)) \
                                    ]
                       ])

train