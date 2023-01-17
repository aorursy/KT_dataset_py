# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('/kaggle/input/used-car-ads/used car for sale - Sheet3.csv')
df
print(df.isnull().sum())
df= df.reset_index().rename(columns={'Year Bought and Km travelled':'year_km'})
df
df[['year','km']]=df.year_km.str.split("-",expand=True,)
df
df=df.drop(['year_km'],axis=1)
import re



df['year']=df['year'].apply(lambda x : int(re.findall('\d+' , x)[0]))

df['Price']=df["Price"].apply(lambda x : int(re.sub(',','',x)[1:]))

df['km']=df['km'].apply(lambda x : re.sub('km' ,'', x))

df['km']=df["km"].apply(lambda x : int(re.sub(',','',x)[1:]))

df
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import re
df.info()
#Identifying how strong each v ariable are related

sns.heatmap(df.corr())
# Maximum discount provided by each Brand

px.scatter(df.groupby(['year']).max()['Price'])
px.scatter(df.groupby(['year']).min()['Price'])