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
#importing necessary library modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import seaborn as sns
#importing the cancer2017 dataset

df=pd.read_csv('../input/cancer2017.csv',engine='python')
#encoding parameter is used since cancer2017.csv contains some non-ASCII value which has to be encoded in utf-8 format

df.head()
#the non-ASCII values are interpretted as unicode block symbols i.e. �. It is replaced with NaN values

df.replace({r'[^\x00-\x7F]+':np.nan}, regex=True, inplace=True)

df.head()
import missingno as msno

msno.matrix(df,color=(0.2,0,0.9))
#cleaning the column names

df.columns = [c.strip() for c in df.columns.values.tolist()]

df.columns = [c.replace(' ','') for c in df.columns.values.tolist()]

df.columns
#Describing the basic statistical details 

df.describe()
#returning information about the dataframe

df.info()
#cleaning cell values

for i in range(0,df.shape[0]):

    for j in range(1,df.shape[1]):

        if ',' in str(df.iloc[i][j]):

            df.iloc[i][j]=df.iloc[i][j].replace(',','')

df.head()
#Converting the columns expect state to numeric values

df=df.apply(pd.to_numeric, errors='ignore')

df.info()
df.head()
#Plotting a data with null or NaN values will lead to data inconsistency. 

#To rectify that we will be filling the missing values with the mean value of the corresponding column as stated in the exploratory data analysis.

y=list(df.columns)

bdf=df.copy()

for col in range(1,len(y)):

    bdf[y[col]].fillna((bdf[y[col]].mean()), inplace=True)

bdf.head()
#creating visualizations with inconsistent dataset

x='State'

i=1

z=["prostate","brain","breast","colon","leukemia","liver","lung","lymphoma","ovary","pancreas"]

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

fig.suptitle('Incomplete Data Set')

for row in ax:

    for col in row:

        col.plot(df[x],df[y[i]])

        i=i+1

i=0

for ax in fig.axes:

    plt.xlabel('States')

    plt.ylabel("no of people affected")

    plt.title(z[i])

    i=i+1

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#after fixing the inconsistent data

i=1

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

fig.suptitle('NaN Filled Data Set')



for row in ax:

    for col in row:

        col.plot(bdf[x],bdf[y[i]])

        i=i+1

i=0

for ax in fig.axes:

    plt.xlabel('States')

    plt.ylabel("no of people affected")

    plt.title(z[i])

    i=i+1

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)



fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#Bar plot

i=1

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

fig.suptitle('Incomplete Data Set')



for row in ax:

    for col in row:

        col.bar(df[x],df[y[i]])

        i=i+1

i=0

for ax in fig.axes:

    plt.xlabel('States')

    plt.ylabel("no of people affected")

    plt.title(z[i])

    i=i+1

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)



fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#After fixing the inconsistent data

i=1

fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))

fig.suptitle('NaN Filled Data Set')



for row in ax:

    for col in row:

        col.bar(bdf[x],bdf[y[i]])

        i=i+1

i=0

for ax in fig.axes:

    plt.xlabel('States')

    plt.ylabel("no of people affected")

    plt.title(z[i])

    i=i+1

    plt.sca(ax)

    plt.xticks(rotation=90)

    plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)



fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
#Correlation between various types of cancer using heatmap

cancertypes=list(df.columns[1:df.shape[1]])

corr = df[cancertypes].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.3f',annot_kws={'size': 12},

           xticklabels= cancertypes, yticklabels= cancertypes,

           cmap= 'summer')
#Scatter plot

#Drawing scatterplots for joint relationships and histograms for univariate distributions:

cancertypes=list(df.columns[1:df.shape[1]])

sns.set(style="ticks", color_codes=True)

sns.set_style='dark'

sns.pairplot(bdf,kind='scatter',palette='husl',hue=None)
#scatter plot

import matplotlib.pyplot as plt

x=df["State"]

y=df["Liver"]

plt.scatter(x,y)

plt.title("state vs liver")

plt.xlabel("states")

plt.ylabel("liver cancer")

plt.xticks(np.arange(1,50,step=2),rotation=90)

plt.show()
# visualization using box plot

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')

ax.set(xlim=(100, 10000))

plt.ylabel('Dependent Variables')

plt.title("Box Plot of Pre-Processed Data Set")

ax = sns.boxplot(data = df, orient = 'h', palette = 'Set2')
#visualization using strip plot

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')

ax.set(xlim=(100, 10000))

plt.ylabel('Dependent Variables')

plt.title("Box Plot of Pre-Processed Data Set")

ax = sns.stripplot(data = df, orient = 'h', palette = 'Set2')
#count plot

sns.set(style="darkgrid")

sns.countplot(x=df['Femalebreast'],data=df)

plt.xticks(np.arange(0,43,step=2),rotation=45)

#joint plot

sns.jointplot(x=df['Liver'], y=df['Femalebreast'], kind="hex", color="#4CB391");
sns.lmplot(x = 'Liver', y = 'Femalebreast', data = df,scatter_kws = {'alpha':0.1,'color':'blue'},line_kws={'color':'red'})