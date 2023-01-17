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
import re

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express  as px
FILEPATH = '/kaggle/input/ontario/ontario-public-sector-salary-2018.csv'
df = pd.read_csv(FILEPATH)
df.head()
df.columns
df.isnull().sum()
len(df)
df = df.drop(['Last Name', 'First Name', 'Calendar Year'], axis = 1)
df.sample(5)
# Clean amount column

df['Salary Paid'] = df['Salary Paid'].apply(lambda x : float(re.sub("[^\d\.]", "", x)))

df['Taxable Benefits'] = df['Taxable Benefits'].apply(lambda x : float(re.sub("[^\d\.]", "", x)))



df['Salary Paid'] = df['Salary Paid'].astype(int)

df['Taxable Benefits'] = df['Taxable Benefits'].astype(int)
df.info()
df.sample(3)
df = df.reset_index()
df.head()
type(df['Sector'].value_counts())
temp_df = pd.DataFrame(df['Sector'].value_counts().head(10)).reset_index()



temp_df
fig = go.Figure(data=[go.Pie(labels=temp_df['index'],

                             values=temp_df['Sector'],

                             hole=.7,

                             title = '% entries by Sector',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     

                     ])

fig.update_layout(title = '% entries by Sector')

fig.show()
temp_df = pd.DataFrame(df['Employer'].value_counts().head(10)).reset_index()



fig = go.Figure(data=[go.Pie(labels=temp_df['index'],

                             values=temp_df['Employer'],

                             hole=.7,

                             title = '% entries by Employer',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     

                     ])

fig.update_layout(title = '% entries by Employer')

fig.show()
temp_df = pd.DataFrame(df['Job Title'].value_counts().head(10)).reset_index()



fig = go.Figure(data=[go.Pie(labels=temp_df['index'],

                             values=temp_df['Job Title'],

                             hole=.7,

                             title = '% entries by Job Title',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     

                     ])

fig.update_layout(title = '% entries by Job Title')

fig.show()
import squarify
def show_treemap(col):

    df_type_series = df.groupby(col)['index'].count().sort_values(ascending = False).head(20)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:10],  # show labels for only first 10 items

                  alpha=.2 )

    plt.title('TreeMap by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap('Sector')