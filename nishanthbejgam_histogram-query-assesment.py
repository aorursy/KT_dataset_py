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
import matplotlib.pyplot as plt
Dataset=pd.read_csv("../input/7210_1.csv")
Dataset.columns
Dataset=Dataset[['colors','prices.amountMin','prices.amountMax']]
Dataset.shape
Dataset.columns
Dataset.head()
Dataset.isnull().sum()
Dataset.tail()
Dataset.colors.unique()
Dataset['average']=(Dataset['prices.amountMin']+Dataset['prices.amountMax'])/2
Dataset.isnull().sum()
Dataset.colors.value_counts().head(10)
PurpleColor = Dataset.query("colors=='Purple'")

BlackColor = Dataset.query("colors=='Black'")

BrownColor = Dataset.query("colors=='Brown'")

WhiteColor = Dataset.query("colors=='White'")

BlueColor = Dataset.query("colors=='Blue'")

SilverColor = Dataset.query("colors=='Silver'")

PinkColor = Dataset.query("colors=='Pink'")

OtherColor = Dataset.query("colors!='Pink' and colors!='Blue' and colors!='Silver'and colors!='White'and colors!='Brown'")
from scipy.stats import ttest_ind

ttest_ind(OtherColor['average'],PinkColor['average'],equal_var=False)
%matplotlib inline
plt.subplots(figsize=(10,5))

plt.hist(BlackColor['average'],bins=100,range=(0,500),color='Black')

plt.hist(BrownColor['average'],bins=100,range=(0,500),color='Brown')

plt.hist(WhiteColor['average'],bins=100,range=(0,500),color='White')

plt.hist(PinkColor['average'],bins=100,range=(0,500),color='Pink')

plt.legend(['Black','Brown','White','Red','Pink'])

plt.ylim(0,300)

plt.show()