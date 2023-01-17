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
df=pd.read_csv("../input/Advertising.csv")
df=df.drop(['Unnamed: 0'],1)
df
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
feature_cols = ['TV', 'radio', 'newspaper']
X = df[feature_cols]
Y = df.sales
lm = LinearRegression()
lm.fit(X, Y)
print(lm.intercept_)
print(lm.coef_)
lm.score(X, Y)

