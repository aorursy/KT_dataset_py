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
data=pd.read_csv("/kaggle/input/voicegender/voice.csv")
print(data.info())



data.label = [1 if each == "male" else 0 for each in data.label]

print(data.info())



y=data.label.values

x = data.drop(["label"],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2, random_state =42)



x_train= x_train.T

x_test= x_test.T

y_train= y_train.T

y_test= y_test.T
#initialize and sigmoid function

#def initialize_weights_and_bias(dimension):

  #  w=np.full((dimension,1),0.01)

  #  b=0.0

  #  return w,b



#def sigmoid(z):

 #   y_head=1/(1+np.exp(-z))

 #   return y_head
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
