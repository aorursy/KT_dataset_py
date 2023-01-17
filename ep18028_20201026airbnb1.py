# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from collections import Counter



pd.set_option('display.max_columns', None)
train_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv', index_col=0)

train_df.head()
test_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv', index_col=0)

test_df.head()
train_df.info()
test_df.info()
Counter(train_df['neighbourhood_group'])
train_df['neighbourhood_group'] = train_df['neighbourhood_group'].map({'Brooklyn': 1,'Manhattan' : 2,'Queens' : 3,'Staten Island' : 4,'Bronx' : 5})

test_df['neighbourhood_group'] = test_df['neighbourhood_group'].map({'Brooklyn': 1,'Manhattan' : 2,'Queens' : 3,'Staten Island' : 4,'Bronx' : 5})
Counter(train_df['room_type'])
train_df['room_type'] = train_df['room_type'].map({'Private room': 1,'Entire home/apt' : 2,'Shared room' : 3})

test_df['room_type'] = test_df['room_type'].map({'Private room': 1,'Entire home/apt' : 2,'Shared room' : 3})
train_df.head()
train_df.info()
test_df.info()
import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt



sns.set_style("darkgrid")

pyplot.figure(figsize=(10, 10))  # 図の大きさを大き目に設定

sns.heatmap(train_df.corr(), square=True, annot=True)  # 相関係数でヒートマップを作成
aaa = ['neighbourhood_group','room_type','minimum_nights','number_of_reviews',]

X_train = train_df[aaa].to_numpy()

y_train = train_df['price'].to_numpy()

X_test = test_df[aaa].to_numpy()
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()

model.fit(X_train, y_train)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df['price'] = p_test

submit_df.to_csv('submission.csv', index=False)