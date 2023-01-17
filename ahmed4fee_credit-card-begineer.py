# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
credit=pd.read_csv("../input/creditcard.csv")

credit.head()

credit.info()

credit.columns.values
#plotting

f,axes=plt.subplots(2,2,figsize=(7,7))

#sns.boxplot(x="Time" , y="Amount" , data=credit, ax=axes[0, 0])

sns.boxplot(x="Class", y="V1", data=credit, ax=axes[0, 0])

sns.boxplot(x="Class", y="V2", data=credit, ax=axes[1, 0])

sns.boxplot(x="Class", y="V3", data=credit, ax=axes[0, 1])

plt.show()

sns.distplot(credit['Amount'],hist=False,rug=True)



sns.regplot(x="Time",y="Amount",data=credit)


