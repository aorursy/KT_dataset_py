import pandas as pd# This Python 3 environment comes with many helpful analytics libraries installed

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
import matplotlib.pyplot as plt
df=pd.read_csv("../input/Poke.csv")
df.head()
df['Name'].unique()
df.Name.value_counts()
colnames = list(df.columns)
predictors = colnames[:4]
target = colnames[4]
# Splitting data into training and testing data set

import numpy as np
# np.random.uniform(start,stop,size) will generate array of real numbers with size = size

df['is_train'] = np.random.uniform(0, 1, len(df))<= 0.75
df['is_train']
train,test = df[df['is_train'] == True],df[df['is_train']==False]
from sklearn.model_selection import train_test_split
train,test = train_test_split(df,test_size = 0.2)
from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
# Accuracy = train

np.mean(train.Name == model.predict(train[predictors]))
# Accuracy = Test

np.mean(preds==test.Name)