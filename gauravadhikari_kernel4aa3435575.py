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
ls
ds=pd.read_csv('../input/mytectra-test/Credit_Card_Expenses.csv')
ds.head()
expDs = ds['CC_Expenses']
expDs.mean()
expDs.mode()
expDs.var()
expDs.min()
expDs.max()
# It is called as a percentile
expDs.quantile(q=0.2)
expDs.skew()
ds.describe()
ds.tail()
ds.corr()
import matplotlib.pyplot as plt 
%matplotlib inline
plt.hist(expDs)
plt.boxplot(expDs)
expDs.describe()
poDs=pd.read_csv("../input/hypothesis-testing/PO_Processing.csv")
poDs.head()
poDs.info()
poDs.head()
from scipy import stats as mystats
PT=poDs['Processing_Time']
mystats.ttest_1samp(PT,40)
poDs.count()
from math import sqrt
def hell(dataSet):
    a=(dataSet.mean()-40)/(dataSet.std()/sqrt(dataSet.count()))
    return a
hell(dataSet=PT)
salesDf=pd.read_csv("../input/twosample-ttest/Sales_Promotion.csv")
salesDf.head()
mystats.ttest_ind(salesDf['Sales_Out1'],salesDf['Sales_Out2'])
salesDf['Sales_Out1'].mean()
salesDf['Sales_Out2'].mean()
# TO find if 
utilisationDf=pd.read_csv("../input/twosample-ttest/Utilization.csv")
utilisationDf.head()
mystats.ttest_ind(utilisationDf['Old'],utilisationDf['New'])
pairedtTest= pd.read_csv("../input/paired-ttest/Tires.csv")
pairedtTest.head()
# this because we are using the two different brands and they do not have any dependency among them unlike the 
# above statement in which old valyes and new values were taken
mystats.ttest_rel(pairedtTest['Brand_1'],pairedtTest['Brand_2'])
# Concusion : As the null hypothesis is accepted thus we connclude that both the brands have mean life
