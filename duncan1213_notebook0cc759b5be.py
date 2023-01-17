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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import sparse

%matplotlib inline
train = pd.read_csv('../input/clicks_train.csv',usecols=['ad_id', 'clicked'])
ad_train_likelehood = train.groupby('ad_id')['clicked'].agg(['count', 'sum', 'mean']).reset_index()
M = train.clicked.mean()

del train
M
ad_train_likelehood['likelihood'] = (ad_train_likelehood['sum'] + 12*M) / (12 + ad_train_likelehood['count'])



test = pd.read_csv("../input/clicks_test.csv")

test = test.merge(ad_train_likelehood, how='left')

test.likelihood.fillna(M, inplace=True)

test.sort_values(['display_id','likelihood'], inplace=True, ascending=False)

subm = test.groupby('display_id')['ad_id'].apply(lambda x: " ".join(map(str,x))).reset_index()

subm.to_csv("subm.csv", index=False)