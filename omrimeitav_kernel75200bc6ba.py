# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy as sp
import sklearn
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()

def train_my_model(market_train_df, news_train_df):
    
    
    return True
def make_my_predictions(market_obs_df, news_obs_df, predictions_template_df):
    
    return True
train_my_model(market_train_df, news_train_df)

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
  predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
  env.predict(predictions_df)

env.write_submission_file()
# Any results you write to the current directory are saved as output.
