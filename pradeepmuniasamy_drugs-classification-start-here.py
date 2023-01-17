import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import folium 

from folium import plugins

plt.style.use("fivethirtyeight")# for pretty graphs





from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)
sample = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

train_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_nscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
train_scored.head()
train_nscored.head()
train_features.head()
test_features.head()
sample.head()
print("Train Features Missing Values:",sum(train_features.isnull().sum().tolist()))

print("Train Scored Misssng Values:",sum(train_scored.isnull().sum().tolist()))

print("Train Non-Scored Misisng Values:",sum(train_nscored.isnull().sum().tolist()))

print("Test Features Misisng Values:",sum(test_features.isnull().sum().tolist()))
values = train_features[train_features.columns[train_features.columns.str.startswith('c-')]].mean().values

range_ = [i for i in range(0,100)]

Mean_df = pd.DataFrame()

Mean_df['range'] = range_

Mean_df['Values'] = values



fig = px.line(Mean_df, x="range", y="Values", title='Average Cell Viability over the columns')

fig.show()
values = train_features[train_features.columns[train_features.columns.str.startswith('c-')]].max().values

range_ = [i for i in range(0,100)]

Mean_df = pd.DataFrame()

Mean_df['range'] = range_

Mean_df['Values'] = values



fig = px.line(Mean_df, x="range", y="Values", title='Maximum Cell Viability over the columns')

fig.show()
values = train_features[train_features.columns[train_features.columns.str.startswith('c-')]].min().values

range_ = [i for i in range(0,100)]

Mean_df = pd.DataFrame()

Mean_df['range'] = range_

Mean_df['Values'] = values



fig = px.line(Mean_df, x="range", y="Values", title='Minimum Cell Viability over the columns')

fig.show()
sns.kdeplot(data=train_features['c-0'], label="Cell 1", shade=True)

sns.kdeplot(data=train_features['c-1'], label="Cell 2", shade=True)

sns.kdeplot(data=train_features['c-2'], label="Cell 3", shade=True)

sns.kdeplot(data=train_features['c-97'], label="Cell 98", shade=True)

sns.kdeplot(data=train_features['c-98'], label="Cell 99", shade=True)

sns.kdeplot(data=train_features['c-99'], label="Cell 100", shade=True)



# Add title

plt.title("Distribution of Viabilty of First and Last 3 cells")
cells = train_features[train_features.columns[train_features.columns.str.startswith('c-')]]

#Plot heatmap

plt.figure(figsize=(12,6))

sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.9)

plt.title('Correlation: Cell viability', fontsize=15, weight='bold')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()