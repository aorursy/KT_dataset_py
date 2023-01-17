# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.bedrooms.value_counts()
df['bedrooms'].loc[df['bedrooms'] >6] = 3
df.bathrooms.value_counts()
df.bathrooms = df.bathrooms.astype(int)
df.bathrooms.loc[df.bathrooms >4] = 2
df.floors.value_counts()
df.floors = df.floors.astype(int)
df.waterfront.value_counts()
df.condition.value_counts()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.zipcode = le.fit_transform(df.zipcode)

df.head()
X = df.drop('price',axis=1)

y = df.price
X = X.drop(['id','date'],axis = 1)
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor

rr = RandomForestRegressor(n_estimators=100,n_jobs=4,random_state=0)
rr.fit(X_train,y_train)

pr2 = rr.predict(X_valid)

mean_absolute_error(y_valid,pr2)
r2_score(y_valid,pr2)