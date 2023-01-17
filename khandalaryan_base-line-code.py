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
df = pd.read_csv('../input/train.csv')
y = df['Volume']

X = df.drop(['Volume','Date'],axis=1)
X.head()
y.head()
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)
test = pd.read_csv('../input/test.csv')
testdf = test.drop(['Date'],axis=1)
testdf.head()
prediction = reg.predict(testdf)
serial = test['Date']

data = {'Date': serial, 'Volume': prediction}

submission = pd.DataFrame(data)

submission.to_csv('Submission.csv', index=False)