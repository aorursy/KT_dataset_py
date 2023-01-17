# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/data_train.csv', index_col=0)
train.head(20)
in_rect = lambda x,y: 3750901.5068<=x<=3770901.5068 and -19268905.6133<=y<=-19208905.6133
train.columns
train['entry_in_rect'] = train[['x_entry', 'y_entry']].apply(lambda x: int(in_rect(*x)), axis=1)

train['exit_in_rect'] = train[['x_exit', 'y_exit']].apply(lambda x: int(in_rect(*x)), axis=1)
train.head(20)
city_centre_x = (3750901.5068+3770901.5068)/2

city_centre_y = (-19268905.6133-19208905.6133)/2

print(city_centre_x, city_centre_y)
train['x_entry_relative'] = train['x_entry'].apply(lambda x: x - city_centre_x)

train['x_exit_relative'] = train['x_exit'].apply(lambda x: x - city_centre_x)

train['y_entry_relative'] = train['y_entry'].apply(lambda y: y - city_centre_y)

train['y_exit_relative'] = train['y_exit'].apply(lambda y: y - city_centre_y)
train.head(20)
train['str_distance'] = train[

    ['x_entry_relative', 'y_entry_relative', 'x_exit_relative', 'y_exit_relative']

].apply(lambda f: math.sqrt((f[2]-f[0])**2 + (f[3]-f[1])**2), axis=1)

asdf = train[

    ['x_entry_relative', 'y_entry_relative', 'x_exit_relative', 'y_exit_relative']

].loc[4]



print(math.sqrt((asdf[2]-asdf[0])**2 + (asdf[3]-asdf[1])**2))
train.head(20)
def time_to_seconds(s):

    assert isinstance(s, str)

    r = tuple(int(x) for x in s.split(':'))

    assert len(r) == 3

    hours, mins, secs = r

    return hours*3600 + mins*60 + secs
time_to_seconds("14:12:34")
train['time_entry_secs'] = train['time_entry'].apply(time_to_seconds)

train['time_exit_secs'] = train['time_exit'].apply(time_to_seconds)
train.head(20)
train['entry_distance_from_centre'] = train[

    ['x_entry_relative', 'y_entry_relative']

].apply(lambda f: math.sqrt(f[0]**2 + f[1]**2), axis=1)



train['exit_distance_from_centre'] = train[

    ['x_exit_relative', 'y_exit_relative']

].apply(lambda f: math.sqrt(f[0]**2 + f[1]**2), axis=1)

train.head(20)
import numpy as np





testset = train[["entry_in_rect","time_entry_secs","time_exit_secs","entry_distance_from_centre","exit_in_rect"]]

x_train = testset[["entry_in_rect","time_entry_secs","time_exit_secs","entry_distance_from_centre"]]

y_train = testset["exit_in_rect"]

testset.columns
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
Model = LogisticRegression()

Model.fit(x_train,y_train)
predictions = Model.predict(x_test)

Accuracy  = Model.score(x_test,y_test)

Accuracy