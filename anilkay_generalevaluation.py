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
happiness=pd.read_csv("../input/world-happiness-report-2019.csv")

happiness.head()
import matplotlib.pyplot as plt #import plotting modules

import seaborn as sns

%matplotlib inline
happiness["Corruption"].describe()
happiness["Healthy life\nexpectancy"].describe()
index=happiness["Healthy life\nexpectancy"].idxmax()

print(happiness.iloc[index,:]) 
index=happiness["Healthy life\nexpectancy"].idxmin() 

print(happiness.iloc[index,:]) 
sns.lmplot(data=happiness,x="Freedom",y="Positive affect")
sns.lmplot(data=happiness,x="Generosity",y="Log of GDP\nper capita")
sns.lmplot(data=happiness,x="Social support",y="Log of GDP\nper capita")
subset=happiness[["Social support","Log of GDP\nper capita"]]

subset.corr()
sns.heatmap(happiness.corr(),annot = True)
sns.lmplot(data=happiness,x="Ladder",y="Log of GDP\nper capita")
sns.lmplot(data=happiness,x="Ladder",y="Social support")