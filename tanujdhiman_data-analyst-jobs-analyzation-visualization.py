# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data.info()
data.isnull().sum()
data.replace(to_replace =-1 , value=np.nan,inplace=True)
data.replace(to_replace ='-1' , value=np.nan,inplace=True)
data.replace(to_replace =-1.0 , value=np.nan,inplace=True)
data.info()
def FindingMissingValues(dataFrame):
    for col in dataFrame.columns:
        print('{0:.2f}% or {1} values are Missing in {2} Column'.format(dataFrame[col].isna().sum()/len(dataFrame)*100,dataFrame[col].isna().sum(),col),end='\n\n')

FindingMissingValues(data)
data.drop(['Easy Apply','Competitors'],1,inplace = True)
data.drop('Job Description',1,inplace=True)
import re
data['Company Name'] = data['Company Name'].apply(lambda x: re.sub(r'\n.*','',str(x)))
data.drop(['Job Title','Salary Estimate','Size','Revenue'],1,inplace = True)
data.head(10)
import plotly.express as px
import seaborn as sns
sns.heatmap(data.corr(), annot =True)
data.info()
fig = px.scatter(data, x="Rating", y="Company Name", 
                 color="Rating", 
                 hover_data=['Headquarters','Location', 'Type of ownership', 'Industry', 'Sector'], 
                 title = "Data Analyst jobs")
fig.show()
import plotly.graph_objects as go
fig2 = go.Figure(data=go.Scatter(
    y = data['Location'],
    mode='markers',
    marker=dict(
        size=16,
        color=data['Rating'], #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )
))
fig2.update_layout(title='Data Analyst Jobs',
                  yaxis_title="Location")
fig2.show()
import matplotlib.pyplot as plt
fig2 = go.Figure(data=go.Scatter(x=data['Headquarters'],
                                y=data['Company Name'],
                                mode='markers',
                                marker_color=data['Rating'],
                                )) 
fig2.update_layout(title='Data Analyst Jobs',
                  xaxis_title="Headquaters",
                  yaxis_title="Company Name")
fig2.show()