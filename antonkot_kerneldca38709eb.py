# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print(os.listdir("../input"))

df = pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.


%matplotlib inline

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

scale = StandardScaler()



columns = df.columns

x_columns = [columns[2], columns[4], columns[12], columns[13]]

xy_columns = [columns[2], columns[4], columns[12], columns[13], columns[15]]

df = df.dropna(subset=xy_columns)

X = df[x_columns]

X[x_columns] = scale.fit_transform(X[x_columns].as_matrix())



Y = df[[columns[15]]]



[columns[15]]



est = sm.OLS(Y, X).fit()

est.summary()







engine = Y.groupby(df['Engine HP']).mean()

engine.plot.bar().scatter(x='Engine HP', y='MSRP')