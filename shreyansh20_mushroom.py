# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/mushrooms.csv')
data.head(10)
Y=data['class']
my_data=data.drop(['class'],axis=1)
my_data.head()
features=my_data.columns.values
%matplotlib inline
ax = my_data['cap-shape'].value_counts().plot(kind='bar',figsize=(14,8),title="Distribution of cap-shape", color='c')
ax.set_ylabel("Frequency")
plt.show()
ax = my_data['habitat'].value_counts().plot(kind='bar',figsize=(14,8),title="Distribution of habitat", color='c')
ax.set_ylabel("Frequency")
plt.show()
for i in features:
    le=preprocessing.LabelEncoder()
    le.fit(my_data[i])
    le.transform(my_data[i])
    my_data[i]=le.fit_transform(my_data[i])
my_data.head()
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Xtr,Xte,Ytr,Yte= train_test_split(my_data,Y,test_size=0.33)
rf.fit(Xtr,Ytr)
pred=rf.predict(Xte)
pred
print(accuracy_score(Yte,pred))
