# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.



from sklearn.linear_model import LinearRegression
train = pd.read_csv('../input/train.csv').dropna

test = pd.read_csv('../input/test.csv').dropna

X_train=train()['x'].values.reshape(-1,1)

y_train=train()['y'].values.reshape(-1,1)

X_test=test()['x'].values.reshape(-1,1)

y_test=test()['y'].values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train,y_train)
yh = model.predict(X_train)
plt.scatter(X_train,y_train)

plt.plot(X_train,yh,c='yellow')
model.score(X_test,y_test)