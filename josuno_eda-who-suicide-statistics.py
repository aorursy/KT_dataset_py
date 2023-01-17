# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read dataframe

dframe = pd.read_csv("../input/who_suicide_statistics.csv")
dframe.head()
dframe.tail()
dframe.info()
# count missing value

dframe.isnull().sum()
dframe['suicides_no_fillna']=dframe.groupby(['country','sex','age']).transform(lambda x: x.fillna(x.mean()))['suicides_no']

dframe['population_fillna']=dframe.groupby(['country','sex','age']).transform(lambda x: x.fillna(x.mean()))['population']
dframe['age']=dframe['age'].str.replace(' years','')

dframe.loc[(dframe['age']=='5-14'),'age']='05-14'
dframe.head()
print("Number of countries:",len(dframe.country.unique()))

print("Year of oldest record:",dframe.year.min())

print("Year of newest record:",dframe.year.max())
import seaborn as sns

from numpy import median

ax = sns.catplot(x="sex", y="suicides_no_fillna",col='age', data=dframe, estimator=median,height=4, aspect=.7,kind='bar')
dframe.groupby(['country','age']).suicides_no_fillna.sum().nlargest(10).plot(kind='barh')
from numpy import sum

sns.catplot('age','suicides_no_fillna',hue='sex',col='year',data=dframe,kind='bar',col_wrap=3,estimator=sum)
%matplotlib inline

temp=dframe.groupby('age').agg('sum').reset_index()

sns.barplot('age','suicides_no_fillna',data=temp)

plt.show()