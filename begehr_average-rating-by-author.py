# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

from catboost import CatBoostRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, ElasticNet

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

import sklearn.model_selection

import math
train_df = pd.read_csv('../input/hse-aml-2020/books_train.csv')

test_df = pd.read_csv('../input/hse-aml-2020/books_test.csv')

sample_df = pd.read_csv('../input/hse-aml-2020/books_sample_submission.csv')
train_df.head()
test_df.head()
author_avg = train_df.groupby("authors").agg({"average_rating": "mean"})

author_avg
submission_df = test_df.join(author_avg, on="authors")

submission_df
submission_df.describe()
avg_rating = train_df["average_rating"].mean()

avg_rating
# fillna with average rating overall books

submission_df = submission_df.fillna(avg_rating)
def write_output(df):

    df[["bookID", "average_rating"]].to_csv("output.csv", index=False)
write_output(submission_df)