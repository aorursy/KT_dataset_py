# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")

df.head()
sns.pairplot(data=df, hue="Species")
from sklearn import svm

modelo = svm.SVC()

Xtrain = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

Ytrain = df[["Species"]]

modelo.fit(Xtrain,Ytrain)
SL = 5.0

SW = 5.0

PL = 1.0

PW = 1.0



print("A espécie é: " + str(modelo.predict(np.array([SL,SW,PL,PW]).reshape(1, -1))))