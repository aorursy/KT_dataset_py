# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/twitter-user-gender-classification/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Amir Shokri

## St code : 9811920009

## E-mail : amirsh.nll@gmail.com
## Naive-Bayes-Report
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/Users/Amirsh.nll/Downloads/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv', encoding ='latin1')

data.info()
data = data.drop(['_unit_id', '_golden', '_unit_state', '_trusted_judgments', '_last_judgment_at', 'profile_yn', 'profile_yn:confidence', 'created', 'description', 'gender_gold', 'link_color', 'profile_yn_gold', 'profileimage', 'sidebar_color', 'text', 'tweet_coord', 'tweet_created', 'tweet_id', 'tweet_location', 'user_timezone', 'gender:confidence', 'gender', 'name'],axis=1)
y = data['tweet_count'].values

y = y.reshape(-1,1)

x_data = data.drop(['tweet_count'],axis = 1)

print(x_data)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x.head(20000)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5,random_state=100)



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train.ravel())

print("Naive Bayes test accuracy: ", nb.score(x_test, y_test))