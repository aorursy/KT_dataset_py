# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.shape
data.info()
data.status.value_counts()
# Let's see the apperance of boys and girls in our dataset

data.gender.value_counts()
sns.set(style='whitegrid')

plt.figure(figsize=(11,9))

sns.countplot(x='status',hue='gender',data=data,palette='pink_r')
gender_placement=data.groupby(['gender','status']).workex.count().reset_index(name='count')

gender_placement['percent']=round(gender_placement['count']/gender_placement.groupby('gender')['count'].transform('sum')*100,2)

gender_placement
data.query('gender=="M" ' ).salary.describe()
data.query('gender=="F"').salary.describe()
plt.figure(figsize=(13,9))

plt.subplot(211)

sns.distplot(data.query('gender=="M"').salary,label='Male salary',norm_hist=True,kde=False)

plt.legend()

plt.title('Distribution of salary for each gender',size=25)

plt.subplot(212)

sns.distplot(data.query('gender=="F"').salary,label='female salary',color='pink',norm_hist=True,kde=False,bins=20)

plt.legend()
plt.figure(figsize=(10,7))

sns.boxplot(x='gender',y='salary',data=data,palette='pink_r')

plt.title('Distribution of salary for each gender',size=25)
data.ssc_p.describe()
sns.set(style='whitegrid')

plt.figure(figsize=(11,9))

sns.boxplot(x='status',y='ssc_p',data=data,palette='afmhot_r')

plt.title('Placement VS Percentage-10th Grade',size=25)
plt.figure(figsize=(11,9))

sns.countplot(x='status',hue='ssc_b',data=data,palette='mako')
sscb_placement=data.groupby(['ssc_b','status']).ssc_p.count().reset_index(name='count')

sscb_placement['percent']=round(sscb_placement['count']/sscb_placement.groupby('ssc_b')['count'].transform('sum')*100,2)

sscb_placement
data.query('ssc_b=="Central" ' ).salary.describe()
data.query('ssc_b=="Others" ' ).salary.describe()
plt.figure(figsize=(13,9))

plt.subplot(211)

sns.distplot(data.query('ssc_b=="Others" ').salary,label='Others board',norm_hist=True,kde=False,bins=20)

plt.legend()

plt.title('Distribution of salary for Board Education',size=25)

plt.subplot(212)

sns.distplot(data.query('ssc_b=="Central" ').salary,label='Central board',color='c',norm_hist=True,kde=False,bins=20)

plt.legend()
plt.figure(figsize=(11,9))

sns.violinplot(x='ssc_b',y='salary',data=data,palette='mako',inner='quartil')
from scipy.stats import spearmanr

sns.jointplot(x='ssc_p',y='hsc_p',data=data,stat_func=spearmanr,kind='reg',height=10)
plt.figure(figsize=(11,9))

sns.boxplot(x='status',y='hsc_p',data=data,palette='afmhot_r')

plt.title('Placement VS Percentage-12th Grade',size=25)
plt.figure(figsize=(11,9))

sns.countplot(x='hsc_s',hue='status',data=data,palette='seismic')
plt.figure(figsize=(11,9))

sns.violinplot(x='status',y='degree_p',data=data,palette='rocket',inner='quartilles')

plt.title('Placement VS Degree Percentage',size=25)
sns.jointplot(x='salary',y='degree_p',data=data,stat_func=spearmanr,kind='reg',height=10,color='c')
plt.figure(figsize=(11,9))

sns.countplot(x='status',hue='degree_t',data=data,palette='mako')
sscb_placement=data.groupby(['degree_t','status']).ssc_p.count().reset_index(name='count')

sscb_placement['percent']=round(sscb_placement['count']/sscb_placement.groupby('degree_t')['count'].transform('sum')*100,2)

sscb_placement
plt.figure(figsize=(11,9))

sns.boxplot(x='degree_t',y='salary',data=data,palette='seismic')

plt.figure(figsize=(11,9))

sns.countplot(x='status',hue='workex',data=data,palette='mako')
Exwork=data.groupby(['workex','status']).ssc_p.count().reset_index(name='count')

Exwork['percent']=round(Exwork['count']/Exwork.groupby('workex')['count'].transform('sum')*100,2)

Exwork