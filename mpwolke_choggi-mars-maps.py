#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS1kmnHeENx_mKm6kGdy7s6WOdXMX4_Ug11WVerQbX2pUqa4jQ7',width=400,height=400)
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

# MapData-Evans-GP-Flatten.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/surviving-mars-maps/MapData-Evans-GP-Flatten.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'MapData-Evans-GP-Flatten.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRdB3_0PRQUfwtMUJSle2hAP8cz74UfY9PwExOns9uHLcjGhbKS',width=400,height=400)
df.head()
df.dtypes
df["Latitude °"].plot.hist()

plt.show()
df["Longitude °"].plot.hist()

plt.show()
df["Latitude °"].plot.box()

plt.show()
df["Longitude °"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['Latitude °'], y_vars='Longitude °', markers="+", size=4)

plt.show()
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='winter')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

sns.boxplot(x='Latitude °', y='Longitude °', data=df, showfliers=False);
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.jointplot(df['Latitude °'],df['Longitude °'],data=df,kind='scatter')
sns.jointplot(df['Latitude °'],df['Longitude °'],data=df,kind='kde',space=0,color='g')
fig=sns.jointplot(x='Latitude °',y='Longitude °',kind='hex',data=df)
g = (sns.jointplot("Latitude °", "Longitude °",data=df, color="r").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='Latitude °',y='Longitude °',data=df)

sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="Latitude °", y="Longitude °", data=df)
fig=sns.lmplot(x="Longitude °", y="Latitude °",data=df)
df.plot.area(y=['Longitude °','Latitude °'],alpha=0.4,figsize=(12, 6));
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Topography)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
sns.countplot(df["Vector Pump"])

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
sns.countplot(df["The Positronic Brain"])

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQYehh4hos4EtKa5j7O1oIXBwm2c0Vs41upknr8D7NQoPEJQAXA',width=400,height=400)
sub = pd.read_csv('/kaggle/input/surviving-mars-maps/MapData-Evans-GP-Flatten.csv', delimiter=',', nrows = nRowsRead)

sub.to_csv('sub.csv', index = False)