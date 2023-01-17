#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSGEb0ryBdVlqpDvlvGaYCs67I42v3RNtLyWEiP0GePvjwsa0aw',width=400,height=400)
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
df = pd.read_csv("../input/2019-ncov/2019-nCoV.csv")
df.head().style.background_gradient(cmap='PRGn')
df.dtypes
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTBs8Doz0jfaEuQeJgRyLfPpYs3gSbV_gtpyRaz7EQzkVCDDaUP',width=400,height=400)
df["病毒"].plot.hist()

plt.show()
df["病毒"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['病毒'], y_vars='肺炎', markers="+", size=4)

plt.show()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='Blues')

plt.show()
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.set(style="dark")

fig=sns.lmplot(x="冠状病毒", y="流行性感冒",data=df)
df.plot.area(y=['冠状病毒','病毒','肺炎','流行性感冒'],alpha=0.4,figsize=(12, 6));
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='冠状病毒',data=df)

    plt.tight_layout()

    plt.show()
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num       
plt.style.use('dark_background')

for col in df[num].drop(['冠状病毒'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('冠状病毒')

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)