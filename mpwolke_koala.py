# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadskoalacsv/koala.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'koala.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.dtypes
print("The number of nulls in each column are \n", df.isna().sum())
df.describe()
sns.distplot(df["TARGET_FID"].apply(lambda x: x**4))

plt.show()
fig, ax =plt.subplots(figsize=(8,6))

sns.scatterplot(x='TARGET_FID', y='Join_Count', data=df)

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
p = df.hist(figsize = (20,20))
sns.regplot(x=df['TARGET_FID'], y=df['END_DATE'])
sns.lmplot(x="TARGET_FID", y="Join_Count", hue="Shape_Area", data=df)
plt.figure(figsize=(10,10))

plt.title('Koala')

ax=sns.heatmap(df.corr(),

               linewidth=2.6,

               annot=True,

               center=1)
sns.countplot(df["Join_Count"])
import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['TARGET_FID', 'Join_Count'], as_index=False).DATE_SEEN.sum()



fig = px.bar(plot_data, x='TARGET_FID', y='DATE_SEEN', color='Join_Count')

fig.update_layout(

    title_text='Koala',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Genre and track

plot_data = df.groupby(['TARGET_FID', 'Join_Count'], as_index=False).DATE_SEEN.sum()



fig = px.line_polar(plot_data, theta='TARGET_FID', r='DATE_SEEN', color='Join_Count')

fig.update_layout(

    title_text='Koala',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Genre and artist

plot_data = df.groupby(['TARGET_FID', 'Join_Count'], as_index=False).DATE_SEEN.sum()



fig = px.line(plot_data, x='TARGET_FID', y='DATE_SEEN', color='Join_Count')

fig.update_layout(

    title_text='Koala',

    height=500, width=1000)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'http://koalahospital.com/k/Portals/0/KoalaHospitalLogoSmall.png',width=400,height=400)