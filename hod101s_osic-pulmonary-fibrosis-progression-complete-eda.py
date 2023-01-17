import os
import pandas as pd 
import numpy as np

import pydicom

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

import matplotlib.pyplot as plt
%matplotlib inline

import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
train_df.head()
train_df.info()
train_df.describe()
train_df.isnull().sum()
def uniques(df):
    print("UNIQUE VALUE STATS")
    print(f"{len(df)} Rows")
    print("Column\t\tUniquevalues")
    for col in df.columns:
        print(f"{col}\t\t{len(df[col].unique())}")
uniques(train_df)
fig = px.histogram(train_df, x="Age")
fig.update_layout(title_text='Age Distribution')
fig.show()
fig = px.histogram(train_df, x="Weeks",marginal="rug")
fig.update_layout(title_text='Weeks Distribution')
fig.show()
fig = px.histogram(train_df, x="Sex")
fig.update_layout(title_text='Sex Counts')
fig.show()
fig = px.histogram(train_df, x="SmokingStatus")
fig.update_layout(title_text='Smoking Status')
fig.show()
fig = px.scatter(train_df, x="FVC", y="Percent")
fig.update_layout(title_text='Percent vs FVC')
fig.show()
fig = px.scatter(train_df, x="FVC", y="Age" , color ="Sex")
fig.update_layout(title_text='Age vs FVC in terms of Sex')
fig.show()
fig = px.scatter(train_df, x="Weeks", y="Percent" , color ="Sex")
fig.update_layout(title_text='Percent vs Weeks')
fig.show()
fig = px.scatter(train_df, x="Weeks", y="FVC" , color ="Sex")
fig.update_layout(title_text='FVC vs Weeks')
fig.show()
fig = px.bar(train_df, y='FVC', x='SmokingStatus')
fig.update_layout(title = 'FVC based of Smoking Status')
fig.show()
fig = px.histogram(train_df, x="SmokingStatus", color='Sex')
fig.update_layout(title_text='Smoking Status')
fig.show()
fig = px.histogram(train_df, x="Age", color='Sex')
fig.update_layout(title_text='Smoking Status')
fig.show()
test_df.head()
test_df.info()
def getHisto(col):
    fig = px.histogram(test_df, x=col)
    fig.update_layout(title_text=col + ' Distribution')
    fig.show()
getHisto("Weeks")
getHisto("Age")
getHisto("Sex")
getHisto("SmokingStatus")
filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
train_df.loc[train_df.Patient == 'ID00007637202177411956430']
os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
