import numpy as np 

import pandas as pd 

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
data.head(5)
rc = data['Role Category'].value_counts().reset_index()

rc.columns = ['Role Category', 'Count']

rc['Percent'] = rc['Count']/rc['Count'].sum() * 100

rc
rc = rc[:10]

rc
rc = data['Role Category'].value_counts().nlargest(n=10)

rc
fig = px.pie(rc, 

       values = rc.values, 

       names = rc.index, 

       title="Top 10 Role Categories", 

       color=rc.values)

fig.update_traces(opacity=0.5,

                  marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5)

fig.update_layout(title_x=0.5)

fig.show()
location = data['Location'].value_counts().nlargest(n=10)

fig = px.bar(y=location.values,

       x=location.index,

       orientation='v',

       color=location.index,

       text=location.values,

       color_discrete_sequence= px.colors.qualitative.Bold)



fig.update_traces(texttemplate='%{text:.2s}', 

                  textposition='outside', 

                  marker_line_color='rgb(8,48,107)', 

                  marker_line_width=1.5, 

                  opacity=0.7)

fig.update_layout(width=800, 

                  showlegend=False, 

                  xaxis_title="City",

                  yaxis_title="Count",

                  title="Top 10 cities by job count")

fig.show()

data1 = data[:10 ] ## taking just 10 records for demo

data1.head(5)

job_exp_top_10 = pd.DataFrame(data['Job Experience Required'].value_counts()).head(10)

plt.figure(figsize=(10,8))

sns.barplot(data = job_exp_top_10, y = job_exp_top_10.index, x = 'Job Experience Required')

plt.title('Top 10 Range of Job Experience Required', size = 18)

plt.ylabel('Range of Job Experience', size = 15)

plt.xlabel('No. of Jobs Advertised', size = 15)

plt.show()