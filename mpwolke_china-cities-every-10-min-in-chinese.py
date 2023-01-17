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

# 安徽省.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/wuhannovelcoronavirus2019/data_new/河南省.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = '河南省.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.dtypes
df["suspectedCount"].plot.hist()

plt.show()
df["confirmedCount"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['suspectedCount'], y_vars='deadCount', markers="+", size=4)

plt.show()
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='Reds')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(12, 4))

sns.boxplot(x='deadCount', y='locationId', data=df, showfliers=False);
fig=sns.lmplot(x="confirmedCount", y="locationId",data=df)
# venn2

from matplotlib_venn import venn2

locationId = df.iloc[:,0]

curedId = df.iloc[:,1]

suspectedCount = df.iloc[:,2]

deadCount = df.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(locationId)-15, len(curedId)-15, 15), set_labels = ('locationId', 'curedCount'))

plt.show()
df.plot.area(y=['locationId','curedCount','suspectedCount','deadCount'],alpha=0.4,figsize=(12, 6));
sns.barplot(x=df['curedCount'].value_counts().index,y=df['curedCount'].value_counts())
#codes from PSVishnu @psvishnu

hospital = [

    'suspectedCount','curedCount']
sns.pairplot(data=df,diag_kind='kde',vars=hospital,hue='locationId')

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.cityName)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()