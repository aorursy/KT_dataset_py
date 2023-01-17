#Importing the libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go
#Importing the datasets

url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

confirmed=pd.read_csv(url,error_bad_lines=False)

death=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',

                 error_bad_lines=False)

recovered=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',

                 error_bad_lines=False)

def country_with_cases(dataset):

    is_province=dataset.loc[dataset['Province/State'].isna()==False]

    is_province=is_province['Country/Region'].unique()

    df=dataset.copy()

    Country=df['Country/Region'].values

    temp=df.drop(columns=['Province/State','Country/Region','Lat','Long'])

    values=temp.values

    cases=[]

    for i in range(0,len(values)):

        cases.append(values[i][values.shape[1]-1])

    new_df=pd.concat([pd.DataFrame(Country),pd.DataFrame(cases)],axis=1)

    new_df.columns=['Country','Cases']

    index=[]

    is_province_sums=[]

    for i in is_province:

        temp=new_df.loc[new_df['Country']==i]

        index.append(temp.index)

        s=np.sum(temp['Cases'])

        is_province_sums.append(s)

    for i in index:

        new_df.drop(i,axis=0,inplace=True)

    countries_with_province=pd.concat([pd.DataFrame(is_province),pd.DataFrame(is_province_sums)],axis=1)

    countries_with_province.columns=['Country','Cases']

    All_country_cases=pd.concat([new_df,countries_with_province],axis=0)

    All_country_cases.reset_index(inplace=True)

    return All_country_cases
confirmed_tree=country_with_cases(confirmed)
def tree_map(df,color_scale,title):

    #The df is the one which only has country and cases

    fig = px.treemap(df,path=['Country'], values='Cases',color='Cases',

                     color_continuous_scale=color_scale,

                     title=title)

    fig.show()
tree_map(confirmed_tree,'amp','Confirmed cases across countries')
recovered_tree=country_with_cases(recovered)

tree_map(recovered_tree,'Greens','Recovery across countries')
death_tree=country_with_cases(death)

tree_map(death_tree,'Reds','Deaths across countries')
def top_10(df):# This function will only work 

    df_descending=df.sort_values(by='Cases', ascending=False)

    df_descending=df_descending.reset_index()

    top=df_descending.iloc[:10 :]

    return top['Country'].values
def rate(df):   

    is_province=df.loc[confirmed['Province/State'].isna()==False]

    is_province=is_province['Country/Region'].unique()

    copy=df.copy()

    final=[]

    index=[]

    for i in is_province:    

        temp=copy.loc[copy['Country/Region']==i]

        index=copy.loc[copy['Country/Region']==i].index

        temp=temp.sum(axis=0)

        final.append(temp)

        copy.drop(index,inplace=True)

    new_df=pd.DataFrame(final)

    new_df['Country/Region']=is_province

    total=pd.concat([copy,new_df],axis=0)

    total.reset_index(inplace=True)

    total.drop(columns=['Province/State'],inplace=True)

    t=pd.melt(total,id_vars=['Country/Region','index','Lat','Long'],var_name="Date", value_name="Value")

    return t
c=rate(confirmed)

fig = px.line(c, x="Date", y="Value", color="Country/Region",

              title='Confirmed cases across countries')

fig.update_layout(showlegend=False)

fig.show()
r=rate(recovered)

fig = px.line(r, x="Date", y="Value", color="Country/Region",

              title='Confirmed cases across countries')

fig.update_layout(showlegend=False)



fig.show()
d=rate(death)

fig = px.line(d, x="Date", y="Value", color="Country/Region",

              title='Deaths across countries')

fig.update_layout(showlegend=False)



fig.show()
top_ten=top_10(confirmed_tree)
def stacked_line_subplots(confirmed,death,recovered,countries):

    subplot_title=[]

    for i in countries:

        subplot_title.append('Cases in {}'.format(i))

    subplot_title=tuple(subplot_title)

    fig = make_subplots(rows=len(countries), cols=1,subplot_titles=subplot_title)

    #countries=['India','US','Yemen','Angola']

    dates=confirmed.columns

    dates=np.delete(dates,[0,2,3])

    dfs=[confirmed,death,recovered]

    row=1

    for i in range(len(countries)):

        value=[]

        for j in dfs:

            temp=j.loc[j['Country/Region']==countries[i]].values

            temp=np.delete(temp,[0,1,2,3])

            value.append(temp)

        if(i==0):

            fig.append_trace(go.Scatter(x=dates,y=value[1],mode='lines',name='Death',

                         line_color='red',stackgroup='covid',legendgroup="group1"),row=row,col=1)

            fig.append_trace(go.Scatter(x=dates,y=value[2],mode='lines',name='Recovered',

                         line_color='green',stackgroup='covid',legendgroup="group2"),row=row,col=1)

            fig.append_trace(go.Scatter(x=dates,y=value[0],mode='lines',name='Confirmed',

                         line_color='blue',stackgroup='covid',legendgroup="group3"),row=row,col=1)

        else:

            fig.append_trace(go.Scatter(x=dates,y=value[1],mode='lines',name='Death',

                         line_color='red',stackgroup='covid', showlegend=False,

                                        legendgroup="group1"),row=row,col=1)

            fig.append_trace(go.Scatter(x=dates,y=value[2],mode='lines',name='Recovered',

                         line_color='green',stackgroup='covid', showlegend=False,

                                       legendgroup="group2"),row=row,col=1)

            fig.append_trace(go.Scatter(x=dates,y=value[0],mode='lines',name='Confirmed',

                         line_color='blue',stackgroup='covid', showlegend=False,

                                       legendgroup="group3"),row=row,col=1)

            



        row+=1

        

        fig.update_layout(height=2000, width=800,

                  title_text="Cases across the top 10 countries")

            

    fig.show()

    
stacked_line_subplots(confirmed,death,recovered,top_ten)
def world_map(df,title,color):

    dates=df.columns

    dates=np.delete(dates,[0,2,3])

    country=df['Country/Region']

    lat=df['Lat']

    long=df['Long']

    transformed=pd.melt(df,id_vars=['Province/State','Country/Region','Lat','Long'],

                   var_name="Date", value_name="Value")

    fig = px.scatter_geo(transformed, lat="Lat",lon="Long", color="Value",

                     hover_name="Country/Region", size=transformed["Value"],

                     animation_frame="Date",

                     projection="natural earth",title=title,

                     color_continuous_scale=color)

    fig.show()
world_map(confirmed,'Confirmed cases with time series','Jet')
world_map(recovered,'Recovered cases with time series','YlGn')
world_map(death,'Death Cases with time series','Burg')
from datetime import date

def weekly_trend(df,number):

    c=rate(df)

    date_values=c['Date'].values

    first_date=date_values[0].split('/')

    first=list(map(int, first_date))

    for i in range(len(date_values)):

        random=date_values[i].split('/')

        random=list(map(int, random))

        delta=date(2020,random[0],random[1])-date(2020,first[0],first[1])

        date_values[i]=delta.days

    c['Date']=date_values

    c['Weeks']=c['Date']//7

    confirmed_week=c.loc[c['Weeks']==number]

    confirmed_week=confirmed_week.sort_values(by='Value',ascending=False)    

    return confirmed_week

    

def top_3_trends(week,week_number):

    last_day=week.loc[week['Date']==((week_number*7)+6)]

    first_day=week.loc[week['Date']==(week_number*7)]

    first_day_values=first_day['Value'].values

    last_day['Value']=last_day['Value']-first_day_values

    last_day=last_day.sort_values(by='Value',ascending=False)

    countries=last_day['Country/Region'].unique()[:3]

    return countries

    
def top_3(week,week_number,subject):

    top_3=top_3_trends(week,week_number)

    week=week.sort_values(by='Date',ascending=True)

    

    subplot_title=[]

    for i in top_3:

        subplot_title.append( '{}'.format(i))

    subplot_title=tuple(subplot_title)

    fig = make_subplots(rows=1, cols=3,subplot_titles=subplot_title)

    

    col=1

    for i in range(0,3):

        temp=week.loc[week['Country/Region']==top_3[i]]

        temp['Value']-=temp.iloc[0,5]

        fig.append_trace(go.Scatter(x=temp['Date'],y=temp['Value'],mode='lines',showlegend=False

                                    ),col=col,row=1)

        

        col+=1

    

    fig.update_layout(title_text="Week {} trends of {} in top 3 countries".format(week_number,subject))

            

    fig.show()

    

    
for i in range(1,22):

    week=weekly_trend(confirmed,i)

    top_3(week,i,'Confirmed Cases')

#The same can be done for Recovered and death cases, but the the kernel becomes to long and redundant