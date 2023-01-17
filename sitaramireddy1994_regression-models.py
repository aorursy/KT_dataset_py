# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.ExcelFile('../input/test.xlsx',)
df1 = pd.read_excel(df, 'Sheet1')
df2 = pd.read_excel(df, 'Sheet2')
train, test = train_test_split(df1, test_size=0.2)
train2, test2 = train_test_split(df2, test_size=0.2)
trainx= train.iloc[: ,1:7]
trainy= train.iloc[: ,0]
trainx2= train2.iloc[: ,1:13]
trainy2= train2.iloc[: ,0]
testx= test.iloc[: ,1:7]
testy= test.iloc[: ,0]
testx2= test2.iloc[: ,1:13]
testy2= test2.iloc[: ,0]
trainx2.shape

model= RandomForestRegressor()
model.fit(trainx2, trainy2)
model.score(testx2,testy2)
model.predict(testx2.iloc[1:5,:])
testy2.iloc[1:5]
output = model.predict(testx2)
output.shape