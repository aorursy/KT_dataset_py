

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import missingno as msno
print(os.listdir("../input"))

#loading training dataset
data=pd.read_csv("../input/train.csv")
#shape of you
data.shape
data.info()
data.describe()
data.isnull().sum()
data=data.drop(columns=['LotFrontage','Alley','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'])
data.isnull().sum()
data=data.dropna()
data.shape
data.isnull().sum()
data.select_dtypes(include=[np.number]).columns
#this gives the numerical features in the data
data.select_dtypes(include=[np.object]).columns
#this gives the categorical features in the dataframe
msno.matrix(data)
#this helps to visuvalise the data
msno.bar(data)
msno.dendrogram(data)
data.skew()
data.kurt()

