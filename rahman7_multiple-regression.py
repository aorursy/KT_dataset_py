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
# import librery:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import dataset:
dataset=pd.read_csv("../input/50_Startups.csv")
dataset.head()
# visualzation:
sns.pairplot(dataset)
sns.countplot(dataset['State'],data=dataset)

sns.countplot(dataset['Administration'],data=dataset)
sns.countplot(dataset['R&D Spend'],data=dataset)
sns.countplot(dataset['Marketing Spend'],data=dataset)
sns.scatterplot(dataset['Profit'],dataset['State'],data=dataset)
# selecte the dataset on dependent and independes:
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
X
y
# encoding the state:
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_3=LabelEncoder()
X[:,3]=labelencoder_X_3.fit_transform(X[:,3])
onehotencoder=OneHotEncoder()
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
X
# spliting the dataset into tarin and test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
# fitting the dataset on the Multiy_linear_regresson:
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
# prediction of new result:
y_pred=regressor.predict(X_test)

y_pred
y_test

