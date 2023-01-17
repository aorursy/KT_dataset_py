import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

data.columns

# data.axes --> rows
# drop unnecessary columns

# axis=1 --> column, axis=0 --> row

# inplace=True --> remove permanently , default --> inplace=False

data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

# convert characters to binary

data.diagnosis=[1 if each == "M" else 0 for each in data.diagnosis]

data.head()
data['diagnosis'].value_counts()
# split the data set 

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)
# normalization x_data (0-1)

x = (x_data - np.min(x_data))/(np.max(x_data)).values

x.head() # show x
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
# import algorithm -- LogisticRegression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
# fit model with LogisticRegression

lr.fit(x_train, y_train) 
# predict and accuracy -- score

lr.score(x_test,y_test)