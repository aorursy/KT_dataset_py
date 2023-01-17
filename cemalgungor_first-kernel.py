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
#download IRIS data set

data = pd.read_csv('../input/Iris.csv')

print(data)
print(data.shape)
print(data.head())
print(data.tail())
print(data.describe())
print(data.info())
import seaborn as sns

ndata=data.drop('Id', axis=1)

sns.pairplot(ndata , hue='Species')
sns.pairplot(ndata,hue="Species" ,markers='*')
sns.pairplot(ndata, hue="Species", palette="husl")
sns.pairplot(ndata, vars=["SepalWidthCm", "SepalLengthCm"] )
sns.pairplot(ndata,x_vars=["SepalWidthCm", "SepalLengthCm"], y_vars=["PetalWidthCm", "PetalLengthCm"])
sns.pairplot(ndata, diag_kind="hist")
sns.pairplot(ndata, diag_kind="kde")
sns.pairplot(ndata, kind="reg")