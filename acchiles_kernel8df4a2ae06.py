# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# import data file into data frame

AppData=pd.read_csv("../input/googleplaystore.csv")

AppData.head()

AppData=AppData.sort_values('Rating',ascending=False)

AppData.head(20)

AppData['Rating'].mean()

AppData[AppData.Rating.isnull()]

AppData['Rating'].fillna(0,inplace=True)

AppData.head(50)

AppData['Rating'].replace(to_replace=[19],value=0,inplace=True)

AppData['Rating'].mean()

sns.lineplot(data=AppData['Rating'])

AppData['Genres'].value_counts().head(20).plot.bar()
