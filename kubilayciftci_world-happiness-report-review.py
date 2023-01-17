# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2017.csv")
data.info()
data.head(10)
data.corr()
f, ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidth=.7,fmt='.1f',ax=ax)
plt.show()
data.columns
data.plot(kind='scatter', x='Happiness.Score',y='Health..Life.Expectancy.',alpha=0.5,color='red')
plt.xlabel('Happines Score')
plt.ylabel('Health-Life Expentancy')
plt.title('Happiness Score-Health/Life Expentancy Scatter Plot')
plt.show()
happiness=data['Happiness.Score']>7
data[happiness]
data[(data['Economy..GDP.per.Capita.']>1.5)&(data['Freedom']>0.5)]