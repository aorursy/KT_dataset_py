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
df1=pd.read_csv('../input/normal-distribution/PO_Processing.csv')
from scipy import stats as mysp
df1.head()
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
mysp.probplot(df1['Processing_Time'],plot=plt)
mysp.stats.normaltest(df1['Processing_Time'])
corrDf=pd.read_csv("../input/correlation/Correlation.csv")
corrDf.head()
plt.scatter(corrDf['Temperature'],corrDf['Vapor_Pressure'])
np.corrcoef(corrDf['Temperature'],corrDf['Vapor_Pressure'])
corrDf.corr()
