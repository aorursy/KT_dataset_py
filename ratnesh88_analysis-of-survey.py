import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

import os
files = os.listdir("../input")
l =len(files)
file = [0]*l
for i in range(l):
    file[i] = pd.read_csv('../input/'+files[i],low_memory=False)
    file[i] = file[i].set_index(file[i].columns[0])
    print(files[i])
    file[i].info()
colmap = file[0]
colmap[30:50]
cols = colmap.index
cols[30:50]
data = file[2]
data.head()
for col in cols[4:11]:
    title =colmap['Survey Question'][col]
    data[col].value_counts()[:20].iplot(kind='bar',title=title+' Distribution',yTitle='Frequency',xTitle=title+'-'+col)
for col in cols[16:25]:
    title =colmap['Survey Question'][col]
    data[col].value_counts()[:20].iplot(kind='bar',title=title+' Distribution',yTitle='Frequency',xTitle=title+'-'+col)
for col in cols[56:70]:
    title =colmap['Survey Question'][col]
    data[col].value_counts()[:20].iplot(kind='bar',title=title+' Distribution',yTitle='Frequency',xTitle=title+'-'+col,color = 'green')
for col in cols[70:85]:
    title =colmap['Survey Question'][col]
    data[col].value_counts()[:20].iplot(kind='bar',title=title+' Distribution',yTitle='Frequency',xTitle=title+'-'+col,color = 'red')
data1 = file[3]
data1.head(30)
