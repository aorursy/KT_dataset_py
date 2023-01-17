# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
plotly.offline.init_notebook_mode()
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

sur_schema = pd.read_csv('../input/SurveySchema.csv')
free_response = pd.read_csv('../input/freeFormResponses.csv')
mcq_response = pd.read_csv('../input/multipleChoiceResponses.csv')
# Any results you write to the current directory are saved as output.
mcq_response['Q6'].unique()
def filter_datascientists(f):
    return f[f['Q6'].isin(['Data Scientist','Data Analyst'])]

ds = filter_datascientists(mcq_response)
ds.dropna()
ms = ds.groupby(['Q8']).count().reset_index('Q8')

cf.go_offline()
series = ds['Q8'].value_counts()
series.iplot(kind='bar', yTitle='Number of respondants', title='Data Scientists/Analysts by Experience in the role')

def draw_pie(ser,title,labels):
    cf.go_offline()
    trace = plotly.graph_objs.Pie(labels=labels, values=ser)
    data = plotly.graph_objs.Data([trace])
    layout = plotly.graph_objs.Layout(title=title,showlegend=False)
    figure=plotly.graph_objs.Figure(data=data,layout=layout)
    plotly.offline.iplot(figure)
labels = ds['Q8'].unique()
draw_pie(series,'Data Scientists/Analysts by Experience in the role',labels)
ds['Q8'].unique()
def new_datascientists(f):
    return f[f['Q8'].isin(['0-1','1-2'])]
new_ds = new_datascientists(ds)
print(new_ds['Q24'].unique())
print('--------------------------')
print(new_ds['Q25'].unique())
ser_exp_writing = new_ds['Q24'].value_counts()
ser_exp_ml = new_ds['Q25'].value_counts()
labels = new_ds['Q24'].unique()
draw_pie(ser_exp_writing,"Experience writing code to analyze data",labels)
draw_pie(ser_exp_ml,"Experience using machine learning methods",labels)
students = new_ds[new_ds['Q24'] == '3-5 years']
students['study'] = students['Q4'] + students['Q5']
cts = students['study'].value_counts()
draw_pie(cts,"Educational background of <br> respondants writing code <br> for data analysis for 3-5 years",students['study'].unique())
students = new_ds[new_ds['Q25'] == '3-4 years']
students['study'] = students['Q4'] + students['Q5']
cts = students['study'].value_counts()
draw_pie(cts,"Educational background of <br> respondants using ML <br> for 3-4 years",students['study'].unique())
ds['Q18'].unique()
new_datascientist_lang = ds['Q24'].value_counts()
draw_pie(new_datascientist_lang,"Language suggestions by Data scientists in early years",ds['Q18'].unique())


