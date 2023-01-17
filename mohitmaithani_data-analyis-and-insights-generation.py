import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import os
from io import StringIO
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
csv_raw = '/kaggle/input/singapore-resident-dataset/raw.csv'
data = pd.read_csv(csv_raw)
data.head()
data.columns= ['c','year','ethnicity group/gender', 'age', 'population']
data.drop('c',axis=1,inplace=True)
data.info()
data.tail()
data['ethnicity group/gender'].unique()
data['age'].unique()
data.to_csv("raw.csv")
data2 = pd.read_csv('/kaggle/input/singapore-resident-dataset/cleaned- extended_datsaset.csv')
plt.figure(figsize=(10,6))
sns.barplot(x='ethnicity group', y='population',hue= 'gender',data=data2)
plt.figtext(.25,.8,'chinese are the largest',fontsize=13,ha='center')
plt.figure(figsize=(40,10))
sns.barplot(x='year', y='population',hue='ethnicity group',data=data2)
# using first dataset where we have total ethnicity groups 
data.tail()
data['ethnicity group/gender'].unique()
g=data.groupby(['year','ethnicity group/gender'],sort=True)
# g.last()
temp =g.get_group((2018,'Total Chinese'))
temp
pd.to_numeric(temp['population']).sum()
import plotly.express as px
# data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data, x='ethnicity group/gender', y='population')
fig.show()
pd.to_numeric(temp['population']).sum()
plt.figure(figsize=(30,6))
sns.scatterplot(x='age', y='population',data=data2)
plt.figure(figsize=(30,10))
sns.lineplot(x='year', y='population',hue='ethnicity group',markers=True, dashes=False, data=data2)

plt.figtext(.25,.83,'chinese take boom over the years',fontsize=13,ha='right')
plt.figtext(.25,.8,'while other and indians ethnicity are the static over the years',fontsize=13,ha='center')
import plotly.express as px
# data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data, x='age', y='population')
fig.show()
plt.figure(figsize=(50,11))
sns.barplot(x='year', y='population',data=data2)