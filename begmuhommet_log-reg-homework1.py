# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
data.info()
data.head(10)
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
data_diognosis = data.diagnosis
data_diognosis.unique()
# conver 1 or 0
y = [1 if i=="B" else 0 for i in data.diagnosis]
# convert list to np.array
y = np.asarray(y)
# delete the diagnosis column
x = data.drop(["diagnosis"], axis=1)
# normalize
x = ((x - np.min(x)) / (np.max(x) - np.min(x))).values
type(x)
# train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# use LogReg
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
# print test train accuracy
print("train accuracy: {} %".format((log_reg.score(x_test, y_test)) * 100))