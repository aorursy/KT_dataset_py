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
dataset = pd.read_csv("../input/kc_house_data.csv")
dataset.head()
Y = dataset['price'].values

Y.shape

Y.shape[0]
X = dataset.iloc[:,3:]

X.shape[1]
X.iloc[:,3:].values
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = LinearRegression()

model.fit(X_train,y_train)
l = model.coef_

print(len(l))
model.intercept_
y_preds = model.predict(X_test)
acc = model.score(X_test,y_test)
print("Score :",round(acc*100,2))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_preds))