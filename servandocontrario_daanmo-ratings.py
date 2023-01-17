# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
ratings = pd.read_csv('../input/the-movies-dataset/ratings.csv')
ratings_small = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
print ('-ratings info-\n')
ratings.info()
print ('-ratings_small info-\n')
ratings_small.info()
print ('\n-ratings_small NaN-\n')
ratings.isnull().sum()

print ('\n-ratings_small NaN-\n')
ratings_small.isnull().sum()
ratings.describe()
ratings_small.describe()
