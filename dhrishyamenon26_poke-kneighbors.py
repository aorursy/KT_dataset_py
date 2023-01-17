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
import pandas as pd
df=pd.read_csv("../input/Pokemon (2).csv")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
x=df.head(20)
x
df.shape
df.info()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("notebook", font_scale=1.1)
sns.set_style("white")
sns.lmplot('Total','Attack',scatter=True, fit_reg=False, data=x, hue='Name')
x.iloc[:,4:11]
x.iloc[:,2:3]
neighbors=KNeighborsClassifier(n_neighbors=5)
a=x.values[:,4:11]
b=x.values[:,2:3]
x.groupby('Type 1').size()
df.info()
a
trainX, testX, trainY, testY = train_test_split(a,b,test_size=0.3,random_state=0)
neighbors.fit(trainX,trainY)
pred=neighbors.predict(testX)
testX
pred
x1= pd.DataFrame(pred, columns=['Prediction'])
x1