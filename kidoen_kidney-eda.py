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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots
df = pd.read_csv("/kaggle/input/ckdisease/kidney_disease.csv")

df.head()
sns.heatmap(df.isnull(),cmap='Purples')
df.drop(columns=['id'],inplace=True)
df.columns[df.dtypes=='object']
df.head()
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df['classification'] = df['classification'].replace(to_replace={1:"Chronic",0:"Non-Chronic"})
df.rename(columns={'classification':'class'},inplace=True)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)

df['dm'] = df['dm'].replace(to_replace={'\tno':"no",'\tyes':"yes",' yes':"yes", '':np.nan})
df.isnull().sum()
df=df.drop(["su","rbc","rc","wc","pot","sod"],axis=1)
df["pcv"]=df["pcv"].fillna(method="ffill")

df.drop(["pc"],axis=1,inplace=True)

df["hemo"]=df["hemo"].fillna(method="ffill")

df.drop(["sg"],axis=1,inplace=True)

df=df.fillna(method="ffill")

#df.drop(["pcc"],axis=1,inplace=True)

df.drop(["ba"],axis=1,inplace=True)

df.drop(["pe"],axis=1,inplace=True)

df.drop(["cad"],axis=1,inplace=True)

df.drop(["ane"],axis=1,inplace=True)

#df.drop(["dm"],axis=1,inplace=True)
df.isnull().sum()
df=df.replace("\t?",31)
print(df.columns)

print(df.shape[1])
df.head()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True)

plt.show()
fig = px.parallel_categories(df, dimensions=['htn','class','dm'],

                color="age", color_continuous_scale=px.colors.sequential.Inferno) # labeling

fig.update_layout(title="Heart Data Parallel Categories Diagram ",title_x=0.5)

fig.show()
fig = px.parallel_categories(df, dimensions=['pcc','class','appet'],

                color="age", color_continuous_scale=px.colors.sequential.Inferno) # labeling

fig.update_layout(title="Heart Data Parallel Categories Diagram ",title_x=0.5)

fig.show()
plt.figure(figsize=(10,8))

sns.distplot(df['bgr'],color='green')

plt.title("bgr Distribution",fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['age'],color='green')

plt.title("Age Distribution",fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['bp'],color='green')

plt.title("bp Distribution",fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['bu'],color='green')

plt.title("bu Distribution",fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['hemo'],color='green')

plt.title("hemo Distribution",fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['pcv'],color='green')

plt.title("pcv Distribution",fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(df['sc'],color='green')

plt.title("sc Distribution",fontsize=16)

plt.show()
net_category=df['dm'].value_counts().to_frame().reset_index().rename(columns={'index':'dm','dm':'count'})

colors=['orange','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['dm'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Diabetes Mellitus Distribution",title_x=0.5)

fig.show()
net_category=df['htn'].value_counts().to_frame().reset_index().rename(columns={'index':'htn','htn':'count'})

colors=['orange','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['htn'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Hypertension Distribution",title_x=0.5)

fig.show()
net_category=df['class'].value_counts().to_frame().reset_index().rename(columns={'index':'class','class':'count'})

colors=['orange','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['class'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Target Distribution",title_x=0.5)

fig.show()
net_category=df['pcc'].value_counts().to_frame().reset_index().rename(columns={'index':'pcc','pcc':'count'})

colors=['orange','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['pcc'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Pus Cells Distribution",title_x=0.5)

fig.show()
net_category=df['appet'].value_counts().to_frame().reset_index().rename(columns={'index':'appet','appet':'count'})

colors=['orange','lightskyblue']

fig = go.Figure([go.Pie(labels=net_category['appet'], values=net_category['count'])])



fig.update_traces(hoverinfo='label+percent',textinfo='percent+label', textfont_size=15,marker=dict(colors=colors,line=dict(color="black",width=3)))





fig.update_layout(title="Appetite Distribution",title_x=0.5)

fig.show()
df.head()
fig = px.scatter_3d(df, x='age', y='hemo', z='bp',

              color='class', size_max=18,

              symbol='class', opacity=0.7)



fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
fig = px.scatter_3d(df, x='age', y='pcv', z='bu',

              color='class', size_max=18,

              symbol='class', opacity=0.7)



fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
df.head(2)
fig = px.scatter(df,x='age',y='bu',template='plotly_dark',color='class')

fig.show()
fig = px.scatter(df,x='age',y='bgr',template='plotly_dark',color='class')

fig.show()
fig = px.scatter(df,x='age',y='hemo',template='plotly_dark',color='class')

fig.show()
fig = px.scatter(df,x='age',y='bp',template='plotly_dark',color='class')

fig.show()
df.head(2)
fig = px.sunburst(df, path=['class','htn','dm','pcc'], values='age', color='age')

fig.update_layout(title="SunBurst Plot",title_x=0.5)

fig.show()
fig = px.density_heatmap(df, x="age", y="bgr", facet_row="class", facet_col="dm")

fig.update_layout()

fig.show()
fig = px.density_heatmap(df, x="age", y="bp", facet_row="class", facet_col="htn")

fig.update_layout()

fig.show()