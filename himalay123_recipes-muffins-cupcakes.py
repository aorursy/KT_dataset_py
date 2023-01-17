# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn import preprocessing,svm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 









# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/recipes_muffins_cupcakes.csv')

print(df)

df.fillna(-99999,inplace=True)

X=np.array(df.drop('Type',1))

y= np.where(df['Type']=='Muffin', 0, 1) 

print(y)

print(len(X),len(y))

print(X)

X=preprocessing.scale(X)

print(X)

print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

frame=svm.SVC(kernel='linear')

frame.fit(X_train,y_train)

accuracy=frame.score(X_test,y_test)

print(accuracy)