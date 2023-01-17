

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier as dc
df = pd.read_csv("/kaggle/input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv")

df.head()
k = df.keys()

for i in k:

    ob = LabelEncoder()

    n = str(i) + '_n'

    df[n] = ob.fit_transform(df[i])

df
X = df.iloc[:,10:-1]

Y = df['V10_n']

X
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)
model = dc()

model.fit(X,Y)
newdata = [2,0,0,0,0,1,0,0,2]
model.predict([newdata])