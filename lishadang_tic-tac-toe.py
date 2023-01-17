# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder ##convert test data into numbers

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dc
df = pd.read_csv("../input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv")

df.head(4)
k = df.keys()

for i in k:

    ob = LabelEncoder()

    n = str(i)+'_n'

    df[n] = ob.fit_transform(df[i])

df.head(4)
X = df.iloc[:,10:-1]

Y = df['V10_n']

X
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.25)
model = dc()

model.fit(X,Y)
newdata = [2,0,0,0,2,1,0,0,2]
model.predict([newdata])