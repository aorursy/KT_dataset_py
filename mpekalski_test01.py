# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/train.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Get first 10000 rows and print some info about columns
train = pd.read_csv('../input/train.csv',
                    dtype={'orig_destination_distance':np.object, 'user_id':np.int32, 'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['user_id','orig_destination_distance','srch_destination_id','is_booking','hotel_cluster'], nrows= 100000)
                    #chunksize=1000000)
aggs = []
print('-'*38)
for chunk in train:
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.',end='')
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
x=train[pd.isnull(train.orig_destination_distance)].user_id.unique()[1:1000]

y=train[(train.user_id.isin(x)) & (pd.isnull(train.orig_destination_distance))]
y.info()
train.info()