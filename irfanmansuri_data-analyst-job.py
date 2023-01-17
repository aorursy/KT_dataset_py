import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot
import datetime as datetime
import plotly.figure_factory as ff
import cufflinks
cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme="pearl", offline=True)
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
data = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data.shape
# to see whether there is any null values
data.isnull().sum()
data.info()
data.describe()
data = data.dropna()
data.drop('Unnamed: 0', axis=1, inplace=True)
data.head()
# Rating 
data['Rating'].head()
data['Company Name'].head()
# Considering the Company Name only to be in Company Name

data['Company Name'] = data['Company Name'].apply(lambda x: x.splitlines()[0])
data.head()
data['Salary Estimate'].unique()
# Let's find out the description of the Company having -1 as min salary
data[data['Salary Estimate']=='-1']
# Lets replace that with the approx value
data[data['Salary Estimate']=='-1'] = data[data['Salary Estimate']=='-1'].replace('-1', '$30K-$43K (Glassdoor est.)')
# Evaluating the Headquarter data

data['Headquarters'].unique()
data[data['Headquarters']=='-1']
data = data.replace('-1', 'Not Given')
data
data['Size'] = data['Size'].replace('Unknown', 'Not Given')
data['Revenue'] = data['Revenue'].replace('Unknown', 'Not Given')
data = data.drop(['Competitors', 'Easy Apply'], axis=1)
# Considering movie rating can't be negative
data = data.replace(-1,0)
# Plotting the Heatmap
import seaborn as sns
Heatmap = sns.heatmap(data.corr(), annot = True)
# Salary Estimation
px.bar(data['Salary Estimate'].value_counts().reset_index(), 
      x='index',
      y="Salary Estimate",
      labels={'index':'Salary', 'Salary Estimate':'Jobs Available'},
      title="Salary Estimation",
      color="Salary Estimate")
# Various Companies offering the Job

px.bar(data[data['Company Name']!='Not Known'], 
       x='Company Name',
       title="Varios Companies Offering the Job")
px.histogram(data[data['Sector']!='Not Known'], x='Sector',
            title='Number of Vacancis in Each Sector')
px.scatter(data, x="Founded", y="Company Name",
            color="Founded",
          title="Year of Establishment")
fig = px.scatter(data, x="Rating", y="Company Name", 
                 color="Rating", 
                 hover_data=['Headquarters','Location', 'Type of ownership', 'Industry', 'Sector', 'Salary Estimate'], 
                 title = "Details of the role, headquarters, salary etc for the Data Analyst Job")
fig.show()
fig = go.Figure(data=go.Scatter(
y=data['Location'],
mode='markers',
marker=dict(
size=12,
color=data['Rating'],
colorscale='Viridis',
showscale=True)))

fig.update_layout(title="Company Size and Job",
                 yaxis_title='Location')
fig.show()
px.bar(data['Job Title'].value_counts().reset_index().head(30), x='index',
      y='Job Title', labels={'index':'Job_Title', 'Job Title':'Ammount of Vacancies'},
      title = "Vacancies Distribution",
      color="Job Title")
%ls "../input"

