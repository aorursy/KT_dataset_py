#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSUDpgpAErvOFZpzrUbtaJjqoistbFuLbwJyttdD-_NO4tWDdhn&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/identifying-influential-bloggers-techcrunch/comments.csv")
df.head().style.background_gradient(cmap='nipy_spectral')
df = pd.DataFrame({'col':[1,1.1,1.2,]})



df1 = pd.get_dummies(df['col']).add_prefix('topic')

print (df1)
df1.corr()
plt.figure(figsize=(10,4))

sns.heatmap(df1.corr(),annot=True,cmap='Greens')

plt.show()
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
plt.style.use('dark_background')

sns.jointplot(df1['topic1.0'],df1['topic1.1'],data=df1,kind='scatter')
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='topic1.0',y='topic1.2',data=df1)
fig=sns.lmplot(x="topic1.0", y="topic1.2",data=df1)
df1.plot.area(y=['topic1.0','topic1.1','topic1.2'],alpha=0.4,figsize=(12, 6));
sns.barplot(x=df1['topic1.0'].value_counts().index,y=df1['topic1.1'].value_counts())
for col in df1.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='topic1.2',data=df1)

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSFo2NP1UsMjYvwX3Q3PHvr90cqd7DAqTm3xfw1jGff1uhWCWlx&usqp=CAU',width=400,height=400)