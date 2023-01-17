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
pd.set_option('display.max_columns',None)

df_comp = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')

df_elec = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-electricity-2013-2016.csv')

df_water = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')

df_boys_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')

df_girls_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')
top_list_boys = df_boys_toilet.sort_values(['All Schools'],ascending=False)

top_list_girls = df_girls_toilet.sort_values(['All Schools'],ascending = False)

top_list_comp = df_comp.sort_values(['All Schools'],ascending=False)

top_list_elec = df_elec.sort_values(['All Schools'],ascending=False)

top_list_water = df_water.sort_values(['All Schools'],ascending=False)
top_list_water.rename(columns={'State/UT':'State_UT','year':'Year'},inplace=True)
color_list = ['magenta','purple','red','green','blue']

df_list = [top_list_comp,top_list_elec,top_list_water,top_list_boys,top_list_girls]

name_list = ['computerStats','ElectrictyStats','WaterStats','BoysWashrooms','GirlsWashrooms']

for i,j,k in zip(df_list,color_list,name_list):

    #print(f' bar plot of <{k}> DataFrame')

    fig = px.bar(data_frame=i,x = 'State_UT',y = 'All Schools',labels={'x':'State_UT','y':'All Schools'},opacity=0.8,color_discrete_sequence=[j],title=k)

    fig.show()
def allplots(df,name,title):

        

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

            'xaxis':{'title': name},

            'barmode':'relative',

            'title': title,

        }

        fig = go.Figure(data = data,layout=layout)



        iplot(fig)



name_list = ['Computers used stats','Electricity used stats','Water Facility stats','Boys Washroom Stats','Girls Washroom Stats']

title_list = ['States contributing towards Computer Schools','Electricity provided in Schools','Water Facility is Schools','Boys Washroom Stats','Girls Washroom Stats']

for i,j,k in zip(df_list,name_list,title_list):

    allplots(i,j,k)
def bubbleplot(df,j,k):

    data = [

        {

            'y':df[j],

            'x':df[k],

            'mode':'markers',

            'marker':{

                'color':df[j],

                'size':df[k],

                'showscale':True,

            },

            "text":df.State_UT

        }

    ]

    iplot(data)
col_list = ['Primary_Only', 'Primary_with_U_Primary',

       'Primary_with_U_Primary_Sec_HrSec', 'U_Primary_Only',

       'U_Primary_With_Sec_HrSec', 'Primary_with_U_Primary_Sec',

       'U_Primary_With_Sec', 'Sec_Only', 'Sec_with_HrSec.', 'HrSec_Only']

l = 0

for df,j in zip(df_list,col_list):

    k = col_list[l+1]

    print(f' bubble plot of {j} and {k} is shown below ↓')

    bubbleplot(df,j,k);l+=1

    #print('='*120)
def charts(df):

    chart = px.pie(df,values='All Schools',names = 'State_UT',height = 600)

    chart.update_traces(textposition = 'inside',textinfo = 'percent+label')

    

    chart.update_layout(title_x = 0.5,

                       geo = dict(showframe = False,

                                 showcoastlines = False))

    chart.show()
name_list = ['computerStats','ElectrictyStats','WaterStats','BoysWashrooms','GirlsWashrooms']

for i,k in zip(df_list,name_list):

    print(f' pie chart of <{k}> is shown below ↓')

    charts(i)

    #print('='*100)