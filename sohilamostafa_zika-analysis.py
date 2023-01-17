import pandas as pd

import numpy as num

import plotly.graph_objects as go

df = pd.read_csv("../input/zika-viruis/paho_who_cases_reported_latest.csv")

df.columns=df.columns.str.lower().str.replace(" ","_")

df.columns=df.columns.str.replace("laboratory_confirmed_cases","confirmed_cases")

df.columns=df.columns.str.replace("country_/_territory","country_or_territory")

df['suspected_cases'] = pd.to_numeric(df['suspected_cases'].str.replace(",", ""), errors='coerce')

df['measure_values'] = pd.to_numeric(df['measure_values'].str.replace(",", ""), errors='coerce')

df.head()
df.info()
df.describe()
nan=df.isnull().sum().reset_index()

non_nan=df.notnull().sum().reset_index()

print(nan,non_nan)
new_df=df.drop(columns=['date',"measure_names",'country_or_territory.1'])

new_df
new_df=new_df.drop_duplicates(subset=['country_or_territory', 'report_epi_week'], keep='first')

new_df
total_values=new_df['confirmed_cases']+new_df['suspected_cases']

new_df['measure_values']=total_values

new_df
new_df['confirmed_percentage']=(new_df['confirmed_cases']/new_df['measure_values'])*100



new_df.fillna(value=0)                                           
new_df['suspected_percentage']=(new_df['suspected_cases']/new_df['measure_values'])*100



new_df.fillna(value=0) 
con_susRate_per_month = pd.pivot_table(new_df, values =['confirmed_percentage','suspected_percentage'],index =['month_of_date']).reset_index()

  

con_susRate_per_month.sort_values(by=['confirmed_percentage','suspected_percentage'], ascending=False)
data=[]

for col in con_susRate_per_month.columns:

    if col!= 'month_of_date':

        data.append(go.Bar(name=col,x=con_susRate_per_month['month_of_date'],y=con_susRate_per_month[col].tolist(),text=round(con_susRate_per_month[col],2),textposition='auto'))

fig = go.Figure(data)

fig.show()
month_names = ['January','February','March','May','July','October','November','December']

table = pd.pivot_table(new_df, values=['suspected_percentage'], index=['month_of_date'],

                    columns=['country_or_territory'], fill_value=0)

table.columns = [column[1] for column in table.columns]

table.index = [month_names.index(x) for x in table.index]

table.sort_index(inplace=True)

table
fig = go.Figure(data=go.Heatmap(

                   z=table.values,

                   y=[month_names[x] for x in table.index],

                   x=table.columns,

                   hoverongaps = False,colorscale = 'blues'))

fig.show()
month_names = ['January','February','March','May','July','October','November','December']

table = pd.pivot_table(new_df, values=['confirmed_percentage'], index=['month_of_date'],

                    columns=['country_or_territory'], fill_value=0)

table.columns = [column[1] for column in table.columns]

table.index = [month_names.index(x) for x in table.index]

table.sort_index(inplace=True)

table
fig = go.Figure(data=go.Heatmap(

                   z=table.values,

                   y=[month_names[x] for x in table.index],

                   x=table.columns,

                   hoverongaps = False,colorscale = 'Reds'))

fig.show()
table = pd.pivot_table(new_df, values =['confirmed_cases','suspected_cases'],index =['country_or_territory'],aggfunc=num.sum).reset_index()

  

table.sort_values(by=['confirmed_cases','suspected_cases'], ascending=False)

data=[]

for col in table.columns:

    if col!= 'country_or_territory':

        data.append(go.Scatter(name=col,x=table['country_or_territory'],y=table[col].tolist()))

fig = go.Figure(data)

fig.show() 
casesTypes_per_week= pd.pivot_table(new_df, values =['confirmed_cases','suspected_cases'],index =['report_epi_week','year_of_date'],aggfunc=num.sum).reset_index()

casesTypes_per_week
data=[]

for col in casesTypes_per_week.columns:

    if col!= 'report_epi_week':

        data.append(go.Scatter(name=col,x=casesTypes_per_week['report_epi_week'],y=casesTypes_per_week[col].tolist(),text=casesTypes_per_week[col],textposition='top center'))

fig = go.Figure(data)

fig.show() 
average_cases_per_year=new_df.groupby('year_of_date',as_index=False)['measure_values'].mean()

average_cases_per_year
fig = go.Figure(data=go.Scatter(x=average_cases_per_year['year_of_date'], y=average_cases_per_year['measure_values']),layout=go.Layout({

"title":"Total accross years",

"xaxis":{"showgrid":False,"dtick":1},

"yaxis":{"showgrid":False,"range":[0, 330]}



}))

fig.show()


average_cases_per_month=new_df.groupby('month_of_date',as_index=False)['measure_values'].mean()



average_cases_per_month
fig = go.Figure(data=go.Scatter(x=average_cases_per_month['month_of_date'], y=average_cases_per_month['measure_values']),layout=go.Layout({

"title":"Total Monthly",

"xaxis":{"showgrid":False,"dtick":1},

"yaxis":{"showgrid":False,"range":[0, 370]}



}))

fig.show()
REW_per_year=pd.crosstab(new_df['report_epi_week'], new_df['year_of_date']).reset_index()

REW_per_year
fig = go.Figure(data=[go.Box(y=REW_per_year,

            boxpoints='all', 

            jitter=0.3, 

            pointpos=-1.8, 

            boxmean=True

              )])



fig.show()