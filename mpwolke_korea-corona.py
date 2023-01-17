#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQyYcGD4vMoJ219chitRVf1sOcz4UbnVGZLhd0MwWKJCx_T-CdW',width=400,height=400)
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

from plotly.offline import iplot

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/korea-corona/korea_corona_confirmed_by_region.csv")
df.head().style.background_gradient(cmap='cividis')
df.dtypes
sns.pairplot(df, x_vars=['광주'], y_vars='경기', markers="+", size=4)

plt.show()
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='RdYlBu')

plt.show()
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.jointplot(df['광주'],df['전남'],data=df,kind='kde',space=0,color='g')
fig=sns.jointplot(x='경남',y='전북',kind='hex',data=df)
g = (sns.jointplot("인천", "제주",data=df, color="r").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
ax= sns.boxplot(x="경북", y="광주", data=df)

ax= sns.stripplot(x="경북", y="광주", data=df, jitter=True, edgecolor="gray")



boxtwo = ax.artists[2]

boxtwo.set_facecolor('yellow')

boxtwo.set_edgecolor('black')

boxthree=ax.artists[1]

boxthree.set_facecolor('red')

boxthree.set_edgecolor('black')

boxthree=ax.artists[0]

boxthree.set_facecolor('green')

boxthree.set_edgecolor('black')



plt.show()
sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="전남", y="경남", data=df)
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df[num].drop(['충북'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('충북')

    plt.tight_layout()

    plt.show()
fig=sns.lmplot(x="세종", y="인천",data=df)
# venn2

from matplotlib_venn import venn2

세종 = df.iloc[:,0]

서울 = df.iloc[:,1]

전북 = df.iloc[:,2]

경남 = df.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(세종)-15, len(서울)-15, 15), set_labels = ('세종', '서울'))

plt.show()
df.plot.area(y=['부산','광주','강원','제주'],alpha=0.4,figsize=(12, 6));
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.lineplot(x=col,y='강원',data=df)

    plt.tight_layout()

    plt.xlabel(col)

    plt.ylabel('강원')

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='제주',data=df)

    sns.pointplot(x=col,y='제주',data=df,color='Black')

    plt.tight_layout()

    plt.show()
trace1 = go.Box(

    y=df["경남"],

    name = '경남',

    marker = dict(color = 'rgb(0,145,119)')

)



trace2 = go.Box(

    y=df["제주"],

    name = '제주',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='경남 and 제주', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))



fig = dict(data=data, layout=layout)

iplot(fig)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS5IAdbh1Zf8p8SESTGvfbQ07YqWdckGNCq81jesCxQJ5oD--cs',width=400,height=400)