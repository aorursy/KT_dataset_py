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
data=pd.read_csv("../input/athlete_events.csv")
data.head()
data.describe()
data.describe(include="object")
#sorting the dataframe according to year
data.sort_values('Year', ascending=True,inplace=True)
data.head()
#filling missing nan of (Age,Height and Weight) with 0
data.update(data[['Age','Height','Weight']].fillna(0))
data.head()
#converting float to int
cols = ['Age', 'Height','Weight']
data[cols] = data[cols].apply(np.int64)
data.head()
data.isnull().sum()
data.shape
data.dtypes
data.Sex.value_counts(normalize=True)
data['Medal'].value_counts(dropna=False)
data['Sport'].nunique()
data.NOC.nunique()
#US has won more medals than any other country, followed by france
data.Team.value_counts().head()
pd.crosstab(data.Year,data.Medal,margins=True)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
fig=plt.figure(figsize=(25,10))
sns.countplot(x="Year",data=data,hue="Medal")
import seaborn as sns
sns.countplot(x="Season",data=data)
#most of the people are in the age group of 20 to 25
sns.distplot(data.Age)
#most of the people are in the height group of 170cm to 185cm
sns.distplot(data.Height)
#most of the people are in the height group of 60kg to 70kg
sns.distplot(data.Weight)
#filling medal nan with unknown
#data.Medal=data.Medal.fillna('unknown')
#data.head()