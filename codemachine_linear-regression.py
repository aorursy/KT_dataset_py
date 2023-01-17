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
from sklearn import preprocessing

sample_submission = pd.read_csv("../input/wustl-517a-sp20-milestone1/sample_submission.csv")

test = pd.read_csv("../input/wustl-517a-sp20-milestone1/test.csv")

train = pd.read_csv("../input/wustl-517a-sp20-milestone1/train.csv")

#separate features and labels

Y_train=train["Horizontal_Distance_To_Fire_Points"]



#data normalization

X_train=train.drop(['ID','Horizontal_Distance_To_Fire_Points'],axis=1)

X_train_norm = preprocessing.normalize(X_train)



X_test= test.drop(['ID'],axis=1)

X_test_norm = preprocessing.normalize(X_test)

from sklearn import linear_model

reg = linear_model.Ridge(alpha=0.5)

reg.fit(X_train_norm,Y_train)

Y_pred = reg.predict(X_test_norm) 

test_pred = {'ID':test['ID'],

            'Horizontal_Distance_To_Fire_Points':Y_pred}

#write csv

test_pred_pd = pd.DataFrame(test_pred)

test_pred_pd.to_csv('test_pred1.csv',index=False)