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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
data.head(5)
data.shape
data.columns
#We will drop 'Unnamed: 0' column because it does not play any important role in the table
new_data = data.drop(['Unnamed: 0'], axis=1)
new_data.head(4)
print(data['Job Title'].value_counts())
#lets see which category of company is demanding higher number of data analyst
print(data['Industry'].value_counts())
new_data['Sector'].value_counts()
data.Headquarters.hist()
data.Rating.hist()
sns.barplot(x=new_data['Company Name'].value_counts()[0:9],y=new_data['Company Name'].value_counts()[0:9].index)
sns.barplot(x=new_data['Job Title'].value_counts()[0:9],y=new_data['Job Title'].value_counts()[0:9].index)
import plotly.express as px
role = new_data['Job Title'].value_counts().nlargest(n=10)
fig = px.pie(role, 
       values = role.values, 
       names = role.index, 
       title="Top 10 Job Titles", 
       color=role.values,
       color_discrete_sequence=px.colors.qualitative.Prism)
fig.update_traces(opacity=0.7,
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_x=0.5)
fig.show()




role = new_data['Salary Estimate'].value_counts().nlargest(n=10)
fig = px.pie(role, 
       values = role.values, 
       names = role.index, 
       title="Top 10 Salary Estimates", 
       color=role.values,
       color_discrete_sequence=px.colors.qualitative.Prism)
fig.update_traces(opacity=0.7,
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_x=0.5)
fig.show()


sns.barplot(x=new_data['Location'].value_counts()[0:10],y=new_data['Location'].value_counts()[0:10].index)
rev = new_data['Revenue'].value_counts().nlargest(n=10)
fig = px.pie(role, 
       values = rev.values, 
       names = rev.index, 
       title="Top 10 revenues", 
       color=rev.values,
       color_discrete_sequence=px.colors.qualitative.Prism)
fig.update_traces(opacity=0.7,
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_x=0.5)
fig.show()
