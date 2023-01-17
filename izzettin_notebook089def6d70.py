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


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#%% read csv



data2=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')





data2.drop(["Time"], axis=1, inplace=True)



print(data2.info())



y2=data2.Class.values

x_data2=data2.drop(["Class"], axis=1)



#%%normalization



x2 = (x_data2 - np.min(x_data2))/(np.max(x_data2) - np.min(x_data2)).values





#%% train test split



from sklearn.model_selection import train_test_split



x_train2, x_test2, y_train2, y_test2 = train_test_split(x2,y2, test_size = 0.1 ,  random_state=42)





print("x_train: ", x_train2.shape)

print("x_test: ", x_test2.shape)

#%%paramete initializa and sigmued function 



def initialize_weights_and_bias(dimensions):

    w = np.full((dimension, 1), 0.01)

    b = 0.0

    return w,b



def sigmoid(z):

    y_head= 1/(1+np.exp(-z))

    return y_head





#%%Logistic regressiin

    

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train2, y_train2)

print("Test accuracy {} ".format(lr.score(x_test2,y_test2)))






