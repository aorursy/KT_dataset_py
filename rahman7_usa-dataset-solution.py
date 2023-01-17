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
# import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dataset:
df=pd.read_csv("../input/USA.csv")
df
df=df.drop('Address',axis=1)
df
# visualization :
sns.pairplot(df)
sns.scatterplot(df['Price'],df['Avg. Area House Age'],data=df)
sns.scatterplot(df['Price'],df['Avg. Area Income'],data=df)
sns.scatterplot(df['Avg. Area Number of Bedrooms'],df['Price'],data=df)
sns.scatterplot(df['Price'],df['Avg. Area Number of Rooms'],data=df)
sns.heatmap(df)
# spliting the dataset into the dependent and independent:
X=df.iloc[:,:-1].values
y=df.iloc[:,4].values
X
y
# spliting the dataset into the train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# feature scaling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# fitting the dataset  into the SVR:
from sklearn.tree import DecisionTreeRegressor 
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)

# prediction of new result:
y_pred=regressor.predict(X_test)
y_pred
y_test
