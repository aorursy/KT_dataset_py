# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

df.head()
print('Shape of Dataset=',df.shape)

df.describe(include='all')
df.columns = ['Hotel', 'Canceled', 'LeadTime', 'ArrivingYear', 'ArrivingMonth', 'ArrivingWeek','ArrivingDate', 'WeekendStay',

              'WeekStay', 'Adults', 'Children', 'Babies', 'Meal','Country', 'Segment', 'DistChannel','RepeatGuest', 'PrevCancel',

              'PrevBook', 'BookRoomType','AssignRoomType', 'ChangeBooking', 'DepositType', 'Agent','Company', 'WaitingDays', 

              'CustomerType', 'Adress','ParkSpace', 'SpecialRequest','Reservation', 'ReservationDate']
def get_cat_con_var(df):

    unique_list = pd.DataFrame([[i,len(df[i].unique())] for i in df.columns])

    unique_list.columns = ['name','uniques']



    universe = set(df.columns)

    cat_var = set(unique_list.name[(unique_list.uniques<=12)      | 

                                   (unique_list.name=='Country')  | 

                                   (unique_list.name=='Agent')                                     

                                  ])

    con_var = universe - cat_var

    

    return cat_var, con_var 





cat_var, con_var = get_cat_con_var(df)



print("Continuous Variables (",len(con_var),")\n",con_var,'\n\n'

      "Categorical Variables(",len(cat_var),")\n",cat_var)
missing_col_list = df.columns[df.isna().sum()>0]

print('Missing data columns =',missing_col_list)

t = pd.DataFrame([[i,df[i].unique(),df[i].isna().sum()] for i in missing_col_list])

t.columns = ['name','unique','missing']

t   
df.loc[df.Children.isna(),'Children'] = 0



df.loc[df.Country.isna(),'Country'] = 'NAA'



# agent and country are ID, cannot be imputed. Impute available/unavailable.

df.loc[df.Agent>0,'Agent']      = 1

df.loc[df.Agent.isna(),'Agent'] = 0



df.loc[df.Company>0,'Company']      = 1

df.loc[df.Company.isna(),'Company'] = 0



print('Remaining Missing Values = ',df.isna().sum().sum())
def print_unique_values(cols):

    for i in cols:

        print(i,df[i].unique())

        

print_unique_values(cat_var)
df.loc[df.Babies    > 8,'Babies']    = 0

df.loc[df.ParkSpace > 5,'ParkSpace'] = 0

df.loc[df.Children  > 8,'Children']  = 0



df[con_var].describe()
df[con_var].describe()
df.loc[df.LeadTime      > 500,'LeadTime'     ]=500

df.loc[df.WaitingDays   >   0,'WaitingDays'  ]=  1

df.loc[df.WeekendStay   >=  5,'WeekendStay'  ]=  5

df.loc[df.Adults        >   4,'Adults'       ]=  4

df.loc[df.PrevBook      >   0,'PrevBook'     ]=  1

df.loc[df.PrevCancel    >   0,'PrevCancel'   ]=  1

df.loc[df.WeekStay      >  10,'WeekStay'     ]= 10

df.loc[df.ChangeBooking >   5,'ChangeBooking']=  5



cat_var = set(list(cat_var) + ['PrevBook','PrevCancel'])

con_var = set(df.columns) - cat_var



df[con_var].describe()
cor_mat = df.corr()

fig, ax = plt.subplots(figsize=(16,6))

sns.heatmap(cor_mat,ax=ax,cmap="YlGnBu",linewidths=0.1)
pie_list = list()

bar_list = list()

line_list = list()



for i in cat_var:

    if len(df[i].unique())<=5:

        pie_list.append(i)

    elif len(df[i].unique())<=12:

        bar_list.append(i)

    else:

        line_list.append(i)

        

print('Features with 5 levels   \n',pie_list,'\n\n',

      'Features with 5-10 levels\n',bar_list,'\n\n',

      'Features with >10 levels \n',line_list)
def get_pie_label_values(df,col):

    temp = pd.DataFrame([[i,df[df[col]==i].shape[0]] for i in df[col].unique()])    

    return temp[0],temp[1] 
def put_into_bucket(df,col,bucket):    

    diff = int(max(df[col])/bucket)

    for i in range(bucket):    

        df.loc[(df[col] > diff*(i)) & (df[col] <= diff*(i+1)),col] = i+1

    df.loc[df[col]==0,col] = 1

    return df



df = put_into_bucket(df,'LeadTime',bucket=5)
# Extraction

new = df['ReservationDate'].str.split('-', n = 2, expand = True) 

df['YearReserve' ]= new[0] 

df['MonthReserve']= new[1] 

df['DateReserve' ]= new[2] 

df.drop(columns=['ReservationDate'],inplace=True)
n_row = 3

n_col = 5

fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'},{'type':'domain'}],

                                           [{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'},{'type':'domain'}],

                                           [{'type':'domain'}, {'type':'domain'},{'type':'domain'}, {'type':'domain'},{'type':'domain'}]],                                           

                   subplot_titles=pie_list,

                   horizontal_spacing = 0.03, vertical_spacing = 0.08)



row = 1

col = 1

x_adr = 0.082

y_adr = 0.85

x_diff = 0.21 # increasing order

y_diff = 0.845 - 0.485 # decreasing order

ls = list()

for i in pie_list:

    labels, values = get_pie_label_values(df,i)    

    fig.add_trace(go.Pie(labels=labels, values=values, name=i),row,col)      # Design Pie Charts          

    ls.append(dict(text=str('<b>'+i+'</b>'), x=x_adr, y=y_adr, font_size=10, showarrow=False)) # Get position of text in Pie-Holes    

    col+=1                                                                   # Get Grid Details

    x_adr+=x_diff

    if col > n_col:

        col =1

        row+=1

        x_adr = 0.082

        y_adr-= y_diff

    

fig.update_traces(hole=0.65, hoverinfo="label+percent+name")    

fig.update_layout(title_text="Visualizing Categorical Variables using <b>Pie charts</b> : (<i>With or less than 5 levels</i>)",

                  annotations=ls,

                  width=1200,height=650,

                  showlegend=False)

fig.show()
n_row = 1

n_col = 5

fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'bar'}, {'type':'bar'},{'type':'bar'},{'type':'bar'},{'type':'bar'}]],                                                   

                   subplot_titles=bar_list,

                   horizontal_spacing = 0.03, vertical_spacing = 0.13)



row = 1

col = 1

for i in bar_list:

    labels, values = get_pie_label_values(df,i)

    #print(labels, values)

    fig.add_trace(go.Bar(y=values),row=row, col=col)

    

    col+=1

    if col > n_col:

        col =1

        row+=1    

    fig.update_layout(annotations=[dict(font_size=10, showarrow=False)])

    

fig.update_layout(title_text="Visualizing Categorical Variables using <b>Bar charts</b>: (<i>Within 5 - 10 levels</i>)",

                  width=1200,height=500,showlegend=False)

fig.show()
ls=list()

for i in line_list:

    for j in df[i].unique():

        ls.append([j,df[df[i]==j].shape[0],i])

ls = pd.DataFrame(ls)

ls.columns = ['column','counts','feature']



ls.sort_values(by='counts',ascending=False,inplace=True)

fig = px.bar(ls[1:50],x='column',y='counts',color='counts',facet_col='feature')

fig.update_layout(title_text="Visualizing Categorical Variables using <b>Bar charts</b> : (<i>More than 10 levels</i>, Top 50 Countries)",

                  width=1150,height=400,showlegend=False)

fig.show()
list_01 = ['Adults', 'WaitingDays', 'LeadTime', 'ChangeBooking', 

            'WeekStay', 'WeekendStay','YearReserve','MonthReserve']



n_row = 2

n_col = 4

fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'bar'},{'type':'bar'},{'type':'bar'},{'type':'bar'}],

                                                   [{'type':'bar'},{'type':'bar'},{'type':'bar'},{'type':'bar'}]],               

                   subplot_titles=list_01,

                   horizontal_spacing = 0.03, vertical_spacing = 0.23)



row = 1

col = 1

for i in list_01:

    labels, values = get_pie_label_values(df,i)

    #print(labels, values)

    fig.add_trace(go.Bar(x=labels,y=values),row=row, col=col)

    

    col+=1

    if col > n_col:

        col =1

        row+=1    

    fig.update_layout(annotations=[dict(font_size=10, showarrow=False)])

    

fig.update_layout(title_text="Visualizing Continuous Variables using <b>Bar charts</b>",width=1200,height=500,showlegend=False)

fig.show()
list_02 = ['DateReserve','ArrivingDate','ArrivingWeek']



n_row = 3

n_col = 1

fig = make_subplots(rows=n_row, cols=n_col, specs=[[{'type':'bar'}],

                                                   [{'type':'bar'}],

                                                   [{'type':'bar'}]],                                                                                    

                   subplot_titles=list_02,

                   horizontal_spacing = 0.03, vertical_spacing = 0.08)



row = 1

col = 1

for i in list_02:

    labels, values = get_pie_label_values(df,i)

    print(i,'=',min(values))

    values = values - min(values)        

    fig.add_trace(go.Bar(x=labels,y=values),row=row, col=col)

    

    col+=1

    if col > n_col:

        col =1

        row+=1    

    fig.update_layout(annotations=[dict(font_size=10, showarrow=False)])

    

fig.update_layout(title_text="Visualizing Continuous Variables using <b>Bar charts</b>",width=1200,height=700,showlegend=False)

fig.show()