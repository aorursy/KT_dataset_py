# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



data_train=pd.read_csv('../input/train.csv')

data_test=pd.read_csv('../input/test.csv')



X_train,X_test,y_train,y_test=train_test_split(data_train.drop(['PAID_NEXT_MONTH', 'ID'], axis=1).values,data_train['PAID_NEXT_MONTH'].values,test_size=.3,random_state=42)



l=LogisticRegression(solver='liblinear')

l.fit(X_train, y_train)



my_submission = pd.DataFrame({'ID': data_test.ID, 'PAID_NEXT_MONTH': l.predict(data_test.drop(['PAID_NEXT_MONTH', 'ID'], axis=1).values)})

my_submission.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.