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
import os

import numpy as np

import pandas as pd

import pickle as pkl

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import math
df = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")

df.drop(columns=['Unnamed: 0'],inplace=True)
df.head()
y = df['price']

X = df.drop(columns=['price'])
## split data as test and train

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
## default values

cut_map = {

    'Fair' : 1,

    'Good' : 2,

    'Very Good': 3,

    'Premium' : 4,

    'Ideal' : 5  

}



clarity_map = {

    'I1' : 1,

    'SI2' : 2,

    'SI1' : 3,

    'VS2' : 4,

    'VS1' : 5,

    'VVS2' : 6,

    'VVS1' : 7,

    'IF' : 8

}



color_map = {

    'D' : 7,

    'E' : 6,

    'F' : 5,

    'G' : 4,

    'H' : 3,

    'I' : 2,

    'J' : 1

}
def preprocess(df):

    df['cut'] = df['cut'].apply(lambda x : cut_map[x])

    df['clarity'] = df['clarity'].apply(lambda x : clarity_map[x])

    df['color'] = df['color'].apply(lambda x : color_map[x])

    #df.drop(columns=['x','y','z'],inplace=True)

    return df



def scaling(df,scaler=None):

    if scaler==None:

        sc = StandardScaler()

        sc.fit(df)

        df = sc.transform(df)

        pkl.dump(sc,open("diamond_scaler.pkl",'wb'))

    else:

        df = scaler.transform(df)

    return df
X_train = preprocess(X_train)
X_train = scaling(X_train)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_train,y_train)
X_test = preprocess(X_test)
X_test = scaling(X_test,pkl.load(open("diamond_scaler.pkl",'rb')))
y_pred = rfr.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error

math.sqrt(mean_squared_error(y_test,y_pred))
plt.scatter(y_test, rfr.predict(X_test))

plt.xlabel("Actual")

plt.ylabel("Predicted")

x_lim = plt.xlim()

y_lim = plt.ylim()

plt.plot(x_lim, y_lim, "k--")

plt.show()