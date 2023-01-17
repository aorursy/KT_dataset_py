# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the data

train = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv")

test = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")

sample_submission  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
print (train.columns)

test.head()
print (test.shape)

print (sample_submission.shape)
submission_metric_map = '../input/bigquery-geotab-intersection-congestion/submission_metric_map.json'

with open(submission_metric_map, 'r') as f:

    data = json.load(f)

data
sample_submission.head(5)