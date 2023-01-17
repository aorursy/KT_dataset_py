# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import rcParams

import missingno as msno

plt.style.use('seaborn-whitegrid')

# Let's ignore warnings for now

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing Data



df=pd.read_csv('../input/udemy-courses/udemy_courses.csv')

df.head()
# Check for missing values



df.isnull().sum()
df['is_paid'].value_counts()
# Dropping a few unnecessary columns



df=df.drop(['course_id','url','published_timestamp'],axis=1)

df.head()
# Convert the is_paid column to 1 or 0 (False = 0,True = 1)



df['is_paid']=df['is_paid'].apply(lambda x: 1 if x is True else 0 )

df.head()
# Seeing the spread of free vs paid courses



fig = plt.figure(figsize=(20,1))

sns.countplot(data=df,y='is_paid')

df['is_paid'].value_counts()
# Best free courses by Subject



df[df['is_paid']==0].groupby('subject')['num_subscribers'].sum().sort_values(ascending=True).plot(kind='barh')
# Best paid courses by Subject



df[df['is_paid']==1].groupby('subject')['num_subscribers'].sum().sort_values(ascending=True).plot(kind='barh')
# Most popular courses by subject



df.groupby('subject')['num_subscribers'].sum().sort_values(ascending=True).plot(kind='barh')
# Most popular courses 



df.groupby('course_title')['num_subscribers'].sum().sort_values(ascending=True).nlargest(10).plot(kind='barh')
# Most popular free courses 



df[df['is_paid']==0].groupby('course_title')['num_subscribers'].sum().sort_values(ascending=True).nlargest(10).plot(kind='barh')
# Most popular PAID courses 



df[df['is_paid']==1].groupby('course_title')['num_subscribers'].sum().sort_values(ascending=True).nlargest(10).plot(kind='barh')
df.head()
# Distribution of subjects



import plotly.express  as px

import plotly.graph_objects as go

temp_df = pd.DataFrame(df['subject'].value_counts()).reset_index()



fig = go.Figure(data=[go.Pie(labels=temp_df['index'],

                             values=temp_df['subject'],

                             hole=.7,

                             title = '% of Courses by Subject',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     

                     ])

fig.update_layout(title='Amount of Courses by Subject')

fig.show()
df['subject'].plot.pie(figsize=(5, 5))