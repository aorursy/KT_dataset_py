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
data = pd.read_csv('../input/Iris.csv')
data.head(20)
data.dtypes
data.describe()
data.count
data.hist(column=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])


data.boxplot(column =['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], grid = False) 

data['SepalWidthCm'].corr(data['SepalLengthCm'])
data['SepalWidthCm'].corr(data['PetalWidthCm'])
data['SepalWidthCm'].corr(data['PetalLengthCm'])
data['PetalWidthCm'].corr(data['PetalLengthCm'])
data['SepalWidthCm'].corr(data['PetalWidthCm'])
data['SepalLengthCm'].corr(data['PetalLengthCm'])
data.isnull()
data.isnull().any()
Q1 = data['SepalWidthCm'].quantile(0.25)

Q3 = data['SepalWidthCm'].quantile(0.75)

IQR = Q3 - Q1

#print(Q1)

#print(Q3)

#df_out = data.loc[(data['SepalWidthCm'] > Q1) & (data['SepalWidthCm'] < Q3)]

#if(True in ((data['SepalWidthCm'] < (Q1 - 1.5 * IQR)) | (data['SepalWidthCm'] > (Q3 + 1.5 * IQR)))):

 #   print()

    

#data[(data['SepalWidthCm'] < (Q1 - 1.5 * IQR)) | (data['SepalWidthCm'] > (Q3 + 1.5 * IQR))]

data.head(10)

#df_out = data.loc[(data['SepalWidthCm'] > (Q1 - 1.5 * IQR)) | (data['SepalWidthCm'] < (Q3 + 1.5 * IQR))]

#data.drop([data.index[0] , data.index[1]])
data[(data['SepalWidthCm'] < (Q1 - 1.5 * IQR)) | (data['SepalWidthCm'] > (Q3 + 1.5 * IQR))]
outlier = data[(data['SepalWidthCm'] < (Q1 - 1.5 * IQR)) | (data['SepalWidthCm'] > (Q3 + 1.5 * IQR))].index 

print(outlier) 

#data.drop(data.index[])

#data.drop(outlier)