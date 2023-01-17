import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


data=pd.read_csv("../input/comptab_2018-01-29 16_00_comma_separated.csv")
data.head()
data.info()
data.describe()
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data.isnull())
data.columns
data

sns.factorplot('App.', data=data, kind='count')
sns.factorplot('Class', data=data, kind='count', size=12, aspect=1.5)
plt.title(' Class Categories',size=21)
m = pd.DataFrame(data=data['Importer'].value_counts().head(10))
m['Region'] = m.index
m.index = range(10)
m
n = pd.DataFrame(data=data['Exporter'].value_counts().head(10))
n['Region'] = n.index
n.index = range(10)
n
sns.barplot(x=m['Region'], y=m['Importer'], data=m, palette='bright')
sns.barplot(x=n['Region'], y=n['Exporter'], data=n, palette='dark')
data['Term'].value_counts().head()
Percentage = (33862/67161)*100
Percentage
