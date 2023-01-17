# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.isnull().sum()
plt.figure(figsize = (20, 8))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Job Title', data = df,

              order = df['Job Title'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20),

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Job Profile')

plt.show()
plt.figure(figsize = (20, 10))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Key Skills', data = df,

              order = df['Key Skills'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext=(-2, -20), 

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Key Sills that is required')

plt.show()
plt.figure(figsize = (20, 8))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Job Experience Required', data = df,

              order = df['Job Experience Required'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20),

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Experience Ranges that is Required')

plt.show()
co = list(df['Job Experience Required'].value_counts().iloc[:10].index)

df['Job Experience Required'] = np.where(df['Job Experience Required'].isin(co),df['Job Experience Required'],

                                       'Others')
import plotly.graph_objects as go

fig = go.Figure(data = [go.Pie(labels = df['Job Experience Required'].value_counts().index,

                             values = df['Job Experience Required'].value_counts(),

                             pull = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

fig.update_traces(textfont_size = 20,

                  marker = dict(line = dict(color='#000000', width = 2)))

fig.show()
plt.figure(figsize = (20, 8))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Role Category', data = df,

              order = df['Role Category'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20), 

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Domains having openings')

plt.show()
co = list(df['Role Category'].value_counts().iloc[:10].index)

df['Role Category'] = np.where(df['Role Category'].isin(co),df['Role Category'],'Others')
fig = go.Figure(data = [go.Pie(labels = df['Role Category'].value_counts().index,

                             values = df['Role Category'].value_counts(),

                             pull = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

fig.update_traces(textfont_size = 20,

                  marker = dict(line = dict(color = '#000000', width = 2)))

fig.show()
plt.figure(figsize = (20, 8))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Location', data = df,

              order = df['Location'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20), 

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Locations for which Jobs have been Posted')

plt.show()
co = list(df['Location'].value_counts().iloc[:10].index)

df['Location'] = np.where(df['Location'].isin(co),df['Location'],'Others')
fig = go.Figure(data = [go.Pie(labels = df['Location'].value_counts().index,

                             values = df['Location'].value_counts(),

                             pull = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

fig.update_traces(textfont_size = 20,

                  marker = dict(line = dict(color = '#000000', width = 2)))

fig.show()
plt.figure(figsize = (20, 10))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Functional Area', data = df,

              order = df['Functional Area'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20), 

                fontsize = 16,color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Functional Areas for which there is openings')

plt.show()
co = list(df['Functional Area'].value_counts().iloc[:10].index)

df['Functional Area'] = np.where(df['Functional Area'].isin(co),df['Functional Area'],'Others')
fig = go.Figure(data = [go.Pie(labels = df['Functional Area'].value_counts().index,

                             values = df['Functional Area'].value_counts(),

                             pull = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

fig.update_traces(textfont_size = 20,

                  marker = dict(line = dict(color = '#000000', width = 2)))

fig.show()
plt.figure(figsize = (20, 10))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Industry', data = df,

              order = df['Industry'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20), 

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Industry in which there is Openings')

plt.show()
co = list(df['Industry'].value_counts().iloc[:10].index)

df['Industry'] = np.where(df['Industry'].isin(co),df['Industry'],'Others')
import plotly.graph_objects as go

fig = go.Figure(data = [go.Pie(labels = df['Industry'].value_counts().index,

                             values = df['Industry'].value_counts(),

                             pull = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

fig.update_traces(textfont_size = 20,

                  marker = dict(line = dict(color = '#000000', width = 2)))

fig.show()
plt.figure(figsize = (20, 8))

sns.set(style = "darkgrid")

ax = sns.countplot(y = 'Role', data = df,

              order = df['Role'].value_counts().iloc[:10].index)

for p in ax.patches:

    ax.annotate(int(p.get_width()), ((p.get_x() + p.get_width()), p.get_y()), xytext = (-2, -20),

                fontsize = 16, color = '#004d00', textcoords = 'offset points', horizontalalignment = 'right')

plt.title('Top 10 Roles for which Jobs are Posted')

plt.show()
co = list(df['Role'].value_counts().iloc[:10].index)

df['Role'] = np.where(df['Role'].isin(co),df['Role'],'Others')
fig = go.Figure(data = [go.Pie(labels = df['Role'].value_counts().index,

                             values = df['Role'].value_counts(),

                             pull = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

fig.update_traces(textfont_size = 20,

                  marker = dict(line = dict(color = '#000000', width = 2)))

fig.show()
from wordcloud import WordCloud
skills = df['Key Skills'].to_list()

skills = [str(s) for s in skills]

skills = [s.strip().lower()  for i in skills for s in i.split("|")]

skills = " ".join(w for w in skills)

wc = WordCloud(width = 2000, height = 1000).generate(skills)

plt.figure(figsize = (16, 8))

plt.imshow(wc)

plt.axis('off')

plt.show()