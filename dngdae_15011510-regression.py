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
from sklearn.neighbors import KNeighborsRegressor



regressor = KNeighborsRegressor(n_neighbors = 15, weights= "distance")



training_data= pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

test_data= pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

submission=pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')



#preprocessing

training_data=training_data.drop('year',axis=1)

test_data=test_data.drop('year', axis=1)





X_train=training_data.iloc[:,0:4]

Y_train=training_data.iloc[:,4]

#print("train_raw=\n", training_data)

#print("test_raw=\n", test_data)



#print("shape of train=", training_data.shape)

#print("shape of test=", test_data.shape)

#print("shape of X=", X_train.shape)

#print("shape of Y=", Y_train.shape)

#print("X=\n", X_train)

#print("Y=\n", Y_train)





regressor.fit(X_train,Y_train)



guesses = regressor.predict(test_data)

#print("guess=\n" , guesses)
submission["Expected"]=guesses

submission.to_csv('./sample_submit.csv', index=False)