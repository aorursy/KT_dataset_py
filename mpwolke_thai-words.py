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

# thwords.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/thaiwords3k/thwords.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'thwords.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
df.dtypes
df["Unnamed: 0"].plot.hist()

plt.show()
nRowsRead = 1000 # specify 'None' if want to read whole file

# thwords.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

d = pd.read_csv('/kaggle/input/thaiwords3k/th_stopwords.csv', delimiter=',', nrows = nRowsRead)

d.dataframeName = 'th_stopwords.csv'

nRow, nCol = d.shape

print(f'There are {nRow} rows and {nCol} columns')

d.head()
d.dtypes
d = d.rename(columns={'Unanmed: 0':'Unnamed'})
d["Unnamed: 0"].plot.hist()

plt.show()
df["Unnamed: 0"].plot.box()

plt.show()
d["Unnamed: 0"].plot.box()

plt.show()
dcorr=d.corr()

dcorr
sns.heatmap(dcorr,annot=True,cmap='winter')

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in d.word)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
def plot_feature(df,col):

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)

    if d[col].dtype == 'int64':

        d[col].value_counts().sort_index().plot()

    else:

        mean = df.groupby(col)['Unnamed: 0'].mean()

        d[col] = d[col].astype('POS_TYPE')

        levels = mean.sort_values().index.tolist()

        d[col].cat.reorder_categories(levels,inplace=True)

        d[col].value_counts().plot()

    plt.xticks(rotation=45)

    plt.xlabel(col)

    plt.ylabel('Counts')

    plt.subplot(1,2,2)

    

    if d[col].dtype == 'int64' or col == 'Unnamed: 0':

        mean = d.groupby(col)['Unnamed: 0'].mean()

        std = d.groupby(col)['Unnamed: 0'].std()

        mean.plot()

        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values, \

                        alpha=0.1)

    else:

        sns.boxplot(x = col,y='Unnamed: 0',data=df)

    plt.xticks(rotation=45)

    plt.ylabel('Unnamed: 0')

    plt.show()
plot_feature(d,'Unnamed: 0')
plt.style.use('ggplot')

d['Unnamed: 0'].value_counts().plot()

plt.show()
d['Unnamed: 0'].mean()
plt.figure(figsize=(10,5))

d['Unnamed: 0'].plot(kind='hist',bins=50)

plt.show()
for col in d.columns:

    plt.figure(figsize=(19,10))

    sns.barplot(x=col,y='Unnamed: 0',data=d)

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.stripplot(x=col,y='Unnamed: 0',data=d,jitter=True,edgecolor='gray',size=10,palette='winter',orient='v')

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.swarmplot(x=col,y='Unnamed: 0',data=d)

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='Unnamed: 0',data=d)

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.residplot(x=col,y='Unnamed: 0',data=d,lowess=True)

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.distplot(d[col],color='red')

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    plt.plot(col,'Unnamed: 0',data=d,color='orange')

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    plt.bar(col,'Unnamed: 0',data=d,color='Orange')

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('Sales')

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.lineplot(x=col,y='Unnamed: 0',data=d)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('Unnamed: 0')

    plt.show()
import scipy.stats as st

for col in d.columns:

    plt.figure(figsize=(18,9))

    st.probplot(d[col],plot=plt)

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='Unnamed: 0',data=d)

    sns.pointplot(x=col,y='Unnamed: 0',data=d,color='Black')

    plt.tight_layout()

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    sns.boxplot(data=d)

    sns.stripplot(data=d,jitter=True,edgecolor='gray')

    plt.tight_layout()

    plt.ylabel('Unnamed: 0')

    plt.show()
for col in d.columns:

    plt.figure(figsize=(18,9))

    plt.scatter(x=col,y='Unnamed: 0',data=d)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('Unnamed: 0')

    plt.axhline(15,color='Black')

    plt.axvline(50,color='Black')

    plt.show()
#for col in d.columns:

 #   plt.figure(figsize=(18,9))

  #  sns.kdeplot(data=d)

   # plt.tight_layout()

    #plt.show()
sns.pairplot(d)

plt.show()
d = d.rename(columns={'Unanmed: 0':'unnamed'})
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.boxplot(d.Unnamed)

plt.subplot(1,2,2)

sns.distplot(d.Unnamed,bins=20)

plt.show()
#q = d.Unnamed.describe()

#print(q)

#IQR    = q['75%'] - q['25%']

#Upper  = q['75%'] + 1.5 * IQR

#Lower  = q['25%'] - 1.5 * IQR

#print("the upper and lower outliers are {} and {}".format(Upper,Lower))
rows =2



cols = 2



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = d.columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.distplot(d[col[index]],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()
rows = 2

cols = 2



fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))



col = d.columns



index = 0



for i in range(rows):

    for j in range(cols):

        sns.regplot(x=d[col[index]],y=d['Unnamed: 0'],ax=ax[i][j])

        index = index + 1

        

plt.tight_layout()