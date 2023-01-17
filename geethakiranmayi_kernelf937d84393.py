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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import gzip
dataframe=pd.read_csv('../input/winequality_red.csv')
dataframe.info()
dataframe.head()

dataframe.describe()
sns.set()
_=plt.hist(dataframe['pH'])
_ = plt.xlabel("samples")
_ = plt.ylabel("ph value")
plt.show()
plt.plot(dataframe['quality'])
plt.show()
sns.distplot(dataframe["alcohol"],bins=20)
#segregating the features and the target variable from the input dataset
target = dataframe['citric acid']
features = dataframe.drop('citric acid',axis=1)
target = dataframe['quality']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,train_size=0.8)


from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
#Feeding the Model
regressor = LinearRegression()
reg_fit = regressor.fit(X_train,y_train)
reg_pred = reg_fit.predict(X_test)
#score of unnormalized data
score_norm = r2_score(y_test,reg_pred)
print(score_norm)
