# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from tflearn import *

# Any results you write to the current directory are saved as output.
#data preprocessing

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

X_train = train.iloc[:,1:].values.astype('float32')

y_train = train.iloc[:,0].values.astype('int')

X_test = test.values.astype('float32')



from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)



X_train.shape[0]
train_x,cv_x,train_y,cv_y = train_test_split(X_train,y_train,test_size=0.20)

train_x.shape,cv_x.shape
