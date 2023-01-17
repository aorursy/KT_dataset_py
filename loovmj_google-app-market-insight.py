# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# path

app_info_path = '/kaggle/input/google-play-store-apps/googleplaystore.csv'

review_path = '/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv'

# license_path = '/kaggle/input/google-play-store-apps/license.txt'
# read data

app_info = pd.read_csv(app_info_path)

review = pd.read_csv(review_path)
# check data type and volum

app_info.info()
review.info()
# null check

app_info.isna().sum()
app_2 = app_info.dropna(axis ='index', how = 'any')
app_2.columns
# run only one time

# adjust 'Installs' values

## delete '+' and ',' and change into numerical type

app_2['Installs'] = app_2['Installs'].str.replace('+', '').str.replace(',', '')
# change values into numerical values

app_2['Rating'] = pd.to_numeric(app_2['Rating'])

app_2['Reviews'] = pd.to_numeric(app_2['Reviews'])

app_2['Installs'] = pd.to_numeric(app_2['Installs'])
# check correlation

sns.heatmap(app_2[['Rating', 'Installs', 'Reviews']].corr(), annot=True, fmt=".3f");
# 

app_2.columns