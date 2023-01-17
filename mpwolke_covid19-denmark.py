#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRQVtatRkrNXTJIQUlxwNEmhvuinJ3FREO43XQFXZoN62kEKrlJ',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/denmark/corona_Denmark .xltx')
df.head().style.background_gradient(cmap='viridis')
df.dtypes
sns.distplot(df["Confirmed Cases"].apply(lambda x: x**4))

plt.show()
sns.countplot(df["Confirmed Cases"])

plt.xticks(rotation=90)

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

for col in df[num].drop(['Confirmed Cases'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('Confirmed Cases')

    plt.tight_layout()

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.factorplot(x=col,y='Deaths',data=df)

    plt.tight_layout()

    plt.show()
for col in df.columns:

    plt.figure(figsize=(18,9))

    sns.barplot(x=col,y='Deaths',data=df)

    sns.pointplot(x=col,y='Deaths',data=df,color='Black')

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQuv928ArbVGZMtIapsIXFZq_uQ0W6CL2l0h7kHNs6Wvefce7z9',width=400,height=400)