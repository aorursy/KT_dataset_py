# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/winemag-data-130k-v2.csv')
data.info()
data.head()
heat = data.corr()
sns.heatmap(heat)
sns.heatmap(heat, annot = True)
plt.show()
data.isnull().sum()
data.price.isnull().sum()
data[(data['price'] >30) & (data['price'] <100)].sample(130).plot.scatter(x='price', y='points')
plt.show()

data.price.plot.hist(bins=13,range=(0,130),figsize=(13,10))
plt.show()
data.columns
data.country.value_counts().head()
plt.subplots(figsize=(13,7))
sns.countplot('country',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7),order=data['country'].value_counts().head(10).index)
plt.xticks(rotation=90)
plt.title('Number Of Wine Review By Country')
plt.show()
data.taster_name.value_counts().head()
plt.subplots(figsize=(17,7))
sns.countplot('taster_name',data=data,palette= 'plasma_r',edgecolor=sns.color_palette('dark',7),order=data['taster_name'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Wine Review By Wine Taster')
plt.show()
data1=data.drop(["Unnamed: 0"],axis=1)

data1.describe()
print(data['variety'].value_counts(dropna=False).head()) # dropna=True means: except nan values
df = data[data.variety.isin(data.variety.value_counts().head().index)]

sns.boxplot(x='variety',y='points',data=df)
plt.show()
sns.violinplot(x='variety',y='points',data=data[data.variety.isin(data.variety.value_counts()[:5].index)])
plt.show()
new_data=data.drop(["Unnamed: 0"],axis=1).head()
new_data
melted_data=pd.melt(frame=new_data,id_vars='country',value_vars=['province','region_1'])
melted_data
data1=data.head(3)
data2=data.tail(3)
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row
data1=data['country'].head()
data2=data['region_1'].head()
conc_data_col=pd.concat([data1,data2],axis=1)
conc_data_col