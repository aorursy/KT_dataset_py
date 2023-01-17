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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

df = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")

df = df.dropna()

df.head()
df['Dataset'] = df['Dataset'].astype("str")

df.rename(columns={"Dataset":"Target"},inplace=True)

fig = px.scatter(df, x='Age', y='Total_Bilirubin',color='Gender',template='plotly_dark')

fig.update_layout(title="Age v/s Total Bilirubin",xaxis_title="Age",yaxis_title="Total_Bilirubin",title_x=0.5)

fig.show()
fig = px.scatter(df, x='Age', y='Albumin_and_Globulin_Ratio',color='Target',template='plotly_dark')

fig.update_layout(title="Age v/s Albumin_and_Globulin_Ratio",xaxis_title="Age",yaxis_title="Albumin_and_Globulin_Ratio",title_x=0.5)

fig.show()
net_category=df['Gender'].value_counts().to_frame().reset_index().rename(columns={'index':'Gender','Gender':'count'})
colors=['royalblue','orange']

fig = go.Figure([go.Pie(labels=net_category['Gender'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Gender Distribution",title_x=0.5)

fig.show()
net_category=df['Target'].value_counts().to_frame().reset_index().rename(columns={'index':'Target','Target':'count'})

colors=['yellow','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['Target'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Target Distribution",title_x=0.5)

fig.show()
sns.pairplot(df,hue='Target')
sns.pairplot(df,hue='Gender')
df.head()
fig = px.scatter_3d(df,x='Age',y='Direct_Bilirubin',z='Total_Bilirubin',color='Gender')

fig.update_layout(title="Age v/s Direct_Bilirubin v/s Total_Bilirubin",title_x=0.5)
fig = px.scatter_3d(df,x='Age',y='Alkaline_Phosphotase',z='Alamine_Aminotransferase',color='Gender')

fig.update_layout(title="Age v/s Alkaline_Phosphotase v/s Alamine_Aminotransferase",title_x=0.5)
fig = px.scatter_3d(df,x='Age',y='Total_Protiens',z='Albumin_and_Globulin_Ratio',color='Target')

fig.update_layout(title="Age v/s Total_Protiens v/s Albumin_and_Globulin_Ratio",title_x=0.5)
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='viridis')

plt.show()
df['Gender'].unique()
maleage = df[df['Gender']=="Male"]['Age']

femaleage = df[df['Gender']=="Female"]['Age']

hist_data = [maleage,femaleage]



group_labels = ['Age Distribution for Male',"Age Distribution for Female"]

colors=['royalblue',"pink"]

fig = ff.create_distplot(hist_data, group_labels,colors=colors) #custom bin_size

fig.update_layout(title="Age Distribution by Gender",title_x=0.5)

fig.show()
df.head()
fig = go.Figure(go.Histogram2dContour(x=df['Total_Protiens'],

        y=df['Albumin']))

fig.update_layout(title='Density of Total_Protiens & Albumin',xaxis_title="Total_Protiens",yaxis_title="Albumin")

fig.show()