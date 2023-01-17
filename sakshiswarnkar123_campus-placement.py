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
df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.isnull().sum()
df.info()
df.describe(include='all')
df.describe(include='all')
for data in [df]:

    data['status']=data['status'].map({'Placed':1, 'Not Placed':0}).astype(int)
df.groupby(['status','gender'])['gender','status'].count()
df.groupby(['gender', 'status'])[['status']].count()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

ax=sns.FacetGrid(data=df, row='gender', col='status', margin_titles=True)

bins=np.linspace(0, 60, 13)

ax.map(plt.hist,"status", color="orange", bins=bins)
sns.set(style='darkgrid')

ax=sns.FacetGrid(data=df,row='ssc_b',col='status',margin_titles=True)

bins=np.linspace(0,60,13)

ax.map(plt.hist,"status",facecolor='indigo',bins=bins)
df[['hsc_b','status']].groupby(['hsc_b']).mean().sort_values(by="status",ascending=False)
sns.set(style='darkgrid')

ax=sns.FacetGrid(data=df,row='workex',col='status',hue='gender',margin_titles=True)

bins=np.linspace(0,60,13)

(ax.map(plt.hist,"status",bins=bins).add_legend())
df['salary'].dropna().plot()
df['salary'].dropna().plot.box()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['gender']=le.fit_transform(df['gender'])

df.tail(20)
sns.catplot(y="degree_p",x="status",hue="gender",kind="bar", data=df)
sns.catplot(y="degree_p",x="status",hue="specialisation",kind="bar", data=df)
from scipy.stats import pearsonr

corr, _ = pearsonr(df['degree_p'], df['hsc_p'])

corr
data = df[df['status']==1]

data[data['specialisation']=='Mkt&HR'].groupby('gender')[['status']].count().plot.bar()