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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
data=pd.read_csv("../input/accidents_2017.csv")
data.head(10)
data.columns
data.isnull().sum()
ax=data.groupby(['Weekday', 'Part of the day'])['Mild injuries'].last()
ax.head(10)
ax=sns.barplot(x="Victims", y="Weekday", data=data )
sns.relplot(x="Victims", y="Vehicles involved", data=data)
data[['Vehicles involved','Victims']][:10]
dis_name=data['District Name'].value_counts()
dis_name
dis_name.plot(kind='bar')
et=data.groupby(['District Name'])['Victims']
et.plot(kind='bar')
day=data.groupby(["Weekday"])["Victims"].count()
day.column=['Weekday', 'Victims']
day.plot(kind="bar")
day=data.groupby(["Part of the day"])["Victims"].count()
day.column=['Part of the day', 'Victims']
day.plot(kind="bar")
sns.heatmap(data.corr())
