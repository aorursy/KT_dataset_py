# Imports:

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

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

# Add other imports here

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


'''

train_my_model(market_train_df, news_train_df)

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
  predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
  env.predict(predictions_df)

env.write_submission_file()
'''



(market_train_df, news_train_df) = env.get_training_data()

print (market_train_df.shape)

print ("head:\n", market_train_df.head())
print ("tail:\n", market_train_df.tail())

print (news_train_df.shape)

print ("head:\n", news_train_df.head())
print ("tail:\n", news_train_df.tail())

# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()

(market_obs_df, news_obs_df, predictions_template_df) = next(days)
print (market_obs_df.shape)
    
print ("head:\n", market_obs_df.head())
print ("tail:\n", market_obs_df.tail())


print (news_obs_df.shape)

print ("head:\n", news_obs_df.head())
print ("tail:\n", news_obs_df.tail())

market_obs_df.head()

print (predictions_template_df.shape)

predictions_template_df.head()
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
    
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)