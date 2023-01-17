# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
read_data = pd.read_csv('../input/train.csv')
read_data.head()
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(read_data, random_state = 55,
                                        shuffle = True)
train_data.columns
#create a target (i.e, the column you want to predict) for model
Y_train = train_data.Survived
#drop label from X
train_data.drop('Survived', axis = 1, inplace = True)
Y_val = test_data.Survived
test_data.drop('Survived', axis = 1, inplace = True)
train_data.columns
train_data.loc[Y_train == 1].Age.mean()
train_data.loc[Y_train == 0].Age.mean()
train_data.loc[Y_train == 1].Fare.mean()
train_data.loc[Y_train == 0].Fare.mean()
def classify(X):
    '''
    a baseline (a really basic) function that 
    ries to predict whether a person survived 
    or not based on their fare
    
    Arguements:
        X: a DataFrame
    
    Returns:
        a pandas Series (kind of like an array) where
        each element represents 
        0 survived
        1 not survived
    '''
    if(X.Fare < 35):
        return 0
    else:
        return 1
Y_preds = test_data.apply(classify, axis = 1)
from sklearn.metrics import accuracy_score
accuracy_score(Y_val, Y_preds)