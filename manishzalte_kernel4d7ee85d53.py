
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/covid19-patient-precondition-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.express as px

import seaborn as sns
from matplotlib import rcParams
import plotly.graph_objects as go

import matplotlib.pyplot as plt

#importing dataset from kaggel
df= pd.read_csv('../input/covid19-patient-precondition-dataset/covid.csv')

df

#visualation 
df.hist(figsize=(14,14))
plt.show()
df.info()
df.corr()
sns.heatmap(df.corr())
df['icu'].unique
#mapping 97, 98 & 99 3 catogories whic is NA
df['icu']=df['icu'].map({1:1,2:2,97:3,99:3})
df.head()
#type of patient who are there in icu
sns.barplot(x="icu", y="patient_type", data=df)
plt.show()
sns.countplot(x='icu',data=df)
#dropping the icu values where value == 3 which are NA's 
indexnames=df[df['icu']==3].index
df.drop(indexnames,inplace=True)
# now only 1 and 2 cateogries are present
sns.countplot(x='icu',data=df)
#This all people died in icu
df['died'] = df['date_died'].apply(lambda x: 'Non-died' if x == 0 else 'died')
df.head()