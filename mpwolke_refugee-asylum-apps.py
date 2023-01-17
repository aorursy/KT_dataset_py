#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTfks4KSehSkOdMjqKwWqroxc1W_BqG4lSuE_T3V5jLxjtyKEHm',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTCdyjH0i7caNfZUa-bSgrHay2sOYo-DJ_aJgEQjvSgPIgba7fA',width=400,height=400)
df = pd.read_csv('../input/refugee-asylum-applications/migr_asyappctza.tsv', sep='\t', header=None)

df.head()
df.dtypes
df = pd.DataFrame({'col':[1,4,7,8,3,6,5,8,9,0]})



df1 = pd.get_dummies(df['col']).add_prefix('topic')

print (df1)

   
df1["topic1"].plot.hist()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRY1cXcndPlz-Vf7u7hM4zK_bA59Xdb3xrzkQ2p3FFH3uaXN2Cn',width=400,height=400)
df1.corr()
plt.figure(figsize=(10,4))

sns.heatmap(df1.corr(),annot=True,cmap='Blues')

plt.show()
plt.figure(figsize=(10,4))

sns.heatmap(df1.corr(),annot=False,cmap='viridis')

plt.show()
plt.figure(figsize=(10,4))

sns.heatmap(df1.corr(),annot=True,cmap='Reds')

plt.show()
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.jointplot(df1['topic1'],df1['topic3'],data=df1,kind='scatter')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS2Luryqca0s7O4bijDCtaBCozH22b7rlFZYF8fl3XyzUoW9_6h',width=400,height=400)
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='topic1',y='topic3',data=df1)
fig=sns.lmplot(x="topic3", y="topic4",data=df1)
df1.plot.area(y=['topic1','topic3','topic4'],alpha=0.4,figsize=(12, 6));
for col in df1.columns:

    plt.figure(figsize=(18,9))

    sns.stripplot(x=col,y='topic1',data=df1,jitter=True,edgecolor='gray',size=10,palette='winter',orient='v')

    plt.tight_layout()

    plt.show()
sns.barplot(x=df1['topic3'].value_counts().index,y=df1['topic3'].value_counts())
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRzUR9JRqF-rmOwMMtPp6uPMA6Y-LAwIHIsmPFVHqIVSxcsDvdS',width=400,height=400)
for col in df1.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='topic3',data=df1)

    plt.tight_layout()

    plt.show()
for col in df1.columns:

    plt.figure(figsize=(18,9))

    sns.residplot(x=col,y='topic3',data=df1,lowess=True)

    plt.tight_layout()

    plt.show()
for col in df1.columns:

    plt.figure(figsize=(18,9))

    sns.lineplot(x=col,y='topic3',data=df1)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('topic3')

    plt.show()
for col in df1.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='topic3',data=df1)

    sns.pointplot(x=col,y='topic3',data=df1,color='Black')

    plt.tight_layout()

    plt.show()
sns.pairplot(df1)

plt.show()
def plot_feature(df1,col):

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)

    if df1[col].dtype == 'int64':

        df1[col].value_counts().sort_index().plot()

    else:

        mean = df1.groupby(col)['topic1'].mean()

        df1[col] = df1[col].astype('category')

        levels = mean.sort_values().index.tolist()

        df1[col].cat.reorder_categories(levels,inplace=True)

        df1[col].value_counts().plot()

    plt.xticks(rotation=45)

    plt.xlabel(col)

    plt.ylabel('Counts')

    plt.subplot(1,2,2)

    

    if df1[col].dtype == 'int64' or col == 'topic1':

        mean = df1.groupby(col)['topic1'].mean()

        std = df1.groupby(col)['topic1'].std()

        mean.plot()

        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values, \

                        alpha=0.1)

    else:

        sns.boxplot(x = col,y='topic1',data=df1)

    plt.xticks(rotation=45)

    plt.ylabel('Sales')

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTgvwlv-t4z3g4zGITctAG6PkIeJzmcg_jI2b9WlcY-oCg3q7Zt',width=400,height=400)
for col in df1:

    plot_feature(df1,col)

    plt.show()
top = ['topic1','topic3','topic4']
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSaNm-Tu_woMH26d7lr7v3ZtSvHJUQvaXnGl-jixioRunFoMnC7',width=400,height=400)