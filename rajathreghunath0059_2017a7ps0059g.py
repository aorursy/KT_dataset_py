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

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv",index_col='id')

df2 = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv",index_col='id')
less_corr = ['b10','b12','b26','b61','b81','b82','b83','b82','b86','b74','b72','b68','b58','b54','b46','b44','b39','b40','b13','b17','time']

df.drop(less_corr,axis=1,inplace=True)

df2.drop(less_corr,axis=1,inplace=True)
X = df.iloc[:,0:82]

Y = df.iloc[:,82]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.01,random_state=42)
scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
rfr = RandomForestRegressor()

rfr.fit(X_train,Y_train)
X_pred = df2.iloc[:,0:82]

X_pred = scaler.transform(X_pred)
Y_pred = rfr.predict(X_pred)
f = open('solution.csv','w')

f.write('id,label')

f.write('\n')

for i,label in enumerate(Y_pred,start=1):

    f.write(str(i)+","+str(label))

    f.write("\n")

f.close()