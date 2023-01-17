# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ml_metrics as metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
!head -n1 ../input/train.csv > ../working/sampletrain.csv

!shuf -n 10000 ../input/train.csv >> ../working/sampletrain.csv
train=pd.read_csv('../working/sampletrain.csv')
train.shape
train["date_time"] = pd.to_datetime(train["date_time"])

train["year"] = train["date_time"].dt.year

train["month"] = train["date_time"].dt.month
t1 = train[((train.year == 2013) | ((train.year == 2014) & (train.month < 8)))]

t2 = train[((train.year == 2014) & (train.month >= 8))]
t2 = t2[t2.is_booking == True]
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)

predictions = [most_common_clusters for i in range(t2.shape[0])]

target = [[l] for l in t2["hotel_cluster"]]

metrics.mapk(target, predictions, k=5)