# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if '.csv' in filename:

            df = pd.read_csv(os.path.join(dirname, filename))

            print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
enc = OneHotEncoder(handle_unknown='ignore')



y = df.pop('price')



X = df.drop(['id','name','host_id','host_name','number_of_reviews','last_review','reviews_per_month'], axis=1)



enc.fit(X['room_type'])

rooms = X.pop('room_type')

rooms = enc.transform(rooms)

print(rooms)

#X[]