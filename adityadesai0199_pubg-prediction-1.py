# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

from sklearn.ensemble import RandomForestRegressor as RFR



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train =  pd.read_csv("../input/train_V2.csv")

print(train.shape)
train = train.dropna(axis='rows')

train = train.drop('matchType',axis=1)
x = train.iloc[:,3:27]

y = train.iloc[:,27]
model = RFR(n_estimators=20,random_state=0)
model.fit(x,y)
test = pd.read_csv("../input/test_V2.csv")

test = test.dropna(axis='rows')

test = test.drop('matchType',axis=1)
x = test.iloc[:,3:27]

y = model.predict(x)

x = test.iloc[:,0]
output = np.vstack((x,y))     # To merge 2 numpy arrays
output = np.transpose(output)
output = pd.DataFrame(output, columns=['Id','winPlacePerc'])
output.to_csv('submission.csv', index=False)