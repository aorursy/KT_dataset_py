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
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns',None)
df_comp = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')
df_comp.head()
top_list = df_comp.sort_values(['All Schools'],ascending=False)
top_list.head(2)
px.bar(data_frame=top_list,x = 'State_UT',y = 'All Schools',labels = {'x':'State and UT','y':'All Schools'},opacity=0.8,color_discrete_sequence=['purple'])
chart = px.pie(data_frame=top_list,values = 'All Schools',names='State_UT',height = 600)
chart.update_traces(textposition = 'inside',textinfo = 'percent+label')

chart.update_layout(title_x = 0.5,
                   geo = dict(showframe = False,
                             showcoastlines = False))
chart.show()
df = df_comp.copy()
df.head(2)
x = df.State_UT

trace_1 = {
    'x':x,
    'y':df.Primary_Only,
    'name':'Primary_Education',
    'type':'bar'
};
trace_2 = {
    'x':x,
    'y':df.Sec_Only,
    'name':'Secondary_Education',
    'type':'bar'
};
trace_3 = {
    'x':x,
    'y':df.HrSec_Only,
    'name':'HigherSecondary',
    'type':'bar',
};
trace_4 = {
    'x':x,
    'y':df.U_Primary_Only,
    'name':'UnderPrimary',
    'type':'bar',
};
data = [trace_1,trace_2,trace_3,trace_4]
layout = {
    'xaxis':{'title':'Computing in Eductaion'},
    'barmode':'relative',
    'title':'Trend Of computing Education in Indian States',
}
fig = go.Figure(data = data,layout=layout)
iplot(fig)
x = df.State_UT

trace_1 = {
    'x':x,
    'y':df.Primary_Only,
    'name':'Primary_Education',
    'type':'bar'
};
trace_2 = {
    'x':x,
    'y':df.Sec_Only,
    'name':'Secondary_Education',
    'type':'bar'
};
trace_3 = {
    'x':x,
    'y':df.HrSec_Only,
    'name':'HigherSecondary',
    'type':'bar',
};
trace_4 = {
    'x':x,
    'y':df.Primary_with_U_Primary,
    'name':'UnderPrimary',
    'type':'bar',
};
trace_5 = {
    'x':x,
    'y': df.Primary_with_U_Primary_Sec,
    'name':'PrimarywithSecondary',
    'type':'bar',
};
trace_6 = {
    'x':x,
    'y':df.Primary_with_U_Primary_Sec_HrSec,
    'name':'Primary with SeniorSecondary',
    'type':'bar'
};
trace_7 = {
    'x':x,
    'y':df.U_Primary_Only,
    'name':'UnderPrimary',
    'type':'bar',
};
trace_8 = {
    'x':x,
    'y':df.U_Primary_With_Sec,
    'name':'UnderPrimaryWithSecondary',
    'type':'bar'
};
trace_9 = {
    'x':x,
    'y':df.U_Primary_With_Sec_HrSec,
    'name':'UnderPrimarywithSeniorSecondary',
    'type':'bar'
};
data = [trace_1,trace_2,trace_3,trace_4,trace_5,trace_6,trace_7,trace_8,trace_9]
layout = {
    'xaxis':{'title':'Computing in Eductaion'},
    'barmode':'relative',
    'title':'Trend Of computing Education in Indian States',
}
fig = go.Figure(data = data,layout=layout)

iplot(fig)



