import dask

import dask.dataframe as dd



import plotly.express as px



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import gc

import numpy as np



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()



from plotly.subplots import make_subplots

import plotly.graph_objects as go
calendar = dd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sales_train_validation = dd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

sample_submission = dd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

sell_prices = dd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")



sales_train_validation = dd.melt(sales_train_validation,id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],var_name='day',value_name='demand')
item_demand = sales_train_validation.groupby('item_id').sum().compute().sort_values(by='demand',ascending=False)[:50].reset_index()

state_demand = sales_train_validation.groupby('state_id').sum().compute().sort_values(by='demand',ascending=False).reset_index()

cat_demand = sales_train_validation.groupby('cat_id').sum().compute().sort_values(by='demand',ascending=False).reset_index()

dept_demand = sales_train_validation.groupby('dept_id').sum().compute().sort_values(by='demand',ascending=False).reset_index()
fig = make_subplots(rows=2, cols=2,subplot_titles=['cat_demand','state_demand','cat_demand,item_demand','dept_demand'])

fig.add_trace(go.Bar(x=item_demand.item_id, y=item_demand.demand),row=2, col=1)

fig.add_trace(go.Bar(x=state_demand.state_id, y=item_demand.demand),row=1, col=2)

fig.add_trace(go.Bar(x=cat_demand.cat_id, y=item_demand.demand),row=1, col=1)

fig.add_trace(go.Bar(x=dept_demand.dept_id, y=item_demand.demand),row=2, col=2)

fig.update_layout(

    title_text="Plots",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
del item_demand,state_demand,cat_demand,dept_demand

gc.collect()
department_id = ['FOODS_3','FOODS_2','FOODS_1','HOUSEHOLD_1','HOUSEHOLD_2','HOBBIES_1','HOBBIES_2']

dept_over_days_demand = {}

for i in department_id:

    dept_over_days_demand[i] = sales_train_validation[sales_train_validation.dept_id==i].groupby('day').demand.sum().compute()

for i in department_id:

    f = dept_over_days_demand[i].reset_index()

    f['day']=f.day.apply(lambda x: x.split('_')[1]).astype('int')

    f = f.sort_values(by=['day']).set_index('day')

    dept_over_days_demand[i] = f
def c_reset_index(dic):

    for i in dic.keys():

        f = dic[i].reset_index()

        f['day']=f.day.apply(lambda x: x.split('_')[1]).astype('int')

        f = f.sort_values(by=['day']).set_index('day')

        dic[i] = f

    return dic
df_list = ['FOODS_3','FOODS_2','FOODS_1','HOUSEHOLD_1','HOUSEHOLD_2','HOBBIES_1','HOBBIES_2']

df_titles = ['FOODS_3','FOODS_2','FOODS_1','HOUSEHOLD_1','HOUSEHOLD_2','HOBBIES_1','HOBBIES_2']

rc=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1)]

fig = make_subplots(rows=3, cols=3,subplot_titles=df_titles)

for i in range(len(df_list)):

    fig = fig.add_trace(go.Scatter(

    x=dept_over_days_demand[df_list[i]].index,

    y=dept_over_days_demand[df_list[i]].demand,

),row=rc[i][0],col=rc[i][1])

fig.update_layout(

    title_text="Plots over different deptartment id",

     title_font_size=30,

    autosize=False,

    showlegend=False,

    width=1000,

    height=500,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
#Less demand days????

less_dlist = ['d_1062','d_1427','d_1792','d_331','d_697']

less_d=calendar[calendar.d.isin(less_dlist)].compute()

less_d
#seperate data by years 

years = [2011,2012,2013,2014,2015]

dic_years = {}

#add in list of days in given year

for i in years:

    dic_years[i] = calendar.loc[calendar.year==i,'d'].compute().values.tolist()

def getbyyear(groupby_list,aggregate,dic_years=dic_years,years=years):

    d_year = {}

    for i in years:

        d_year[i] = sales_train_validation.loc[sales_train_validation.day.isin(dic_years[i])].groupby(groupby_list).agg({'demand':aggregate}).compute()

    return d_year
#year wise trend over all products on days

year_trend = getbyyear(['day'],'sum')

year_trend_mean = getbyyear(['day'],'mean')

year_trend_std = getbyyear(['day'],'std')



year_trend = c_reset_index(year_trend)

year_trend_mean = c_reset_index(year_trend_mean)

year_trend_std = c_reset_index(year_trend_std)


fig = go.Figure(data=go.Scatter(x=[str(i)for i in years], y=[year_trend[i].sum().values[0] for i in years]))

fig.update_layout(

    title_text="Demand over years",

     title_font_size=30,

    autosize=False,

    showlegend=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
yearly = year_trend[2011]

yearly_mean = year_trend_mean[2011]

yearly_std = year_trend_std[2011]

for i in years[1:]:

    yearly = yearly.append(year_trend[i])

    yearly_mean = yearly_mean.append(year_trend_mean[i])

    yearly_std = yearly_std.append(year_trend_std[i])
daily_trend = pd.merge(pd.merge(yearly,yearly_mean,on='day'),yearly_std,on='day')

daily_trend.columns = ['sum','mean','std']
fig = make_subplots(rows=2, cols=1,subplot_titles=['Mean','Std'])

fig = fig.add_trace(go.Scatter(

    x=daily_trend.index,

    y=daily_trend['mean'],

),row=1,col=1)

fig.add_trace(go.Scatter(

    x=daily_trend.index,

    y=daily_trend['std'],

),row=2,col=1)

fig.update_layout(

    title_text="Mean and Std over all days over all Items",

     title_font_size=30,

    autosize=False,

    showlegend=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
def make_summary(gby=None,columns=None):

    daily_trend_item = getbyyear(gby,'sum')

    daily_trend_mean_item = getbyyear(gby,'mean')

    daily_trend_std_item = getbyyear(gby,'std')

    daily_trend_item = c_reset_index(daily_trend_item)

    daily_trend_mean_item = c_reset_index(daily_trend_mean_item)

    daily_trend_std_item = c_reset_index(daily_trend_std_item)



    daily = daily_trend_item[2011]

    daily_mean = daily_trend_mean_item[2011]

    daily_std = daily_trend_std_item[2011]

    for i in years[1:]:

        daily = daily.append(daily_trend_item[i])

        daily_mean = daily_mean.append(daily_trend_mean_item[i])

        daily_std = daily_std.append(daily_trend_std_item[i])

    daily_item_merged = pd.merge(pd.merge(daily,daily_mean,on=gby),daily_std,on=gby)

    daily_item_merged.columns = columns#['state_id','sum','mean','std']

    return daily_trend_item,daily_trend_mean_item,daily_trend_std_item,daily_item_merged
dept_sum,dept_mean,dept_std,department = make_summary(['day','dept_id'],columns=['dept_id','sum','mean','std'])
fig = px.line(department.reset_index(), x="day", y="sum", color="dept_id",

              line_group="dept_id", hover_name="dept_id")

fig.show()
fig = go.Figure()

for i in department.dept_id.value_counts().index:

    fig.add_trace(go.Histogram(x=department.loc[department.dept_id==i,'sum'],name=i))

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Histogram over states",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
fig = px.box(department, x="dept_id", y="sum",color="dept_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Boxplot over dept_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()
cat_sum,cat_mean,cat_std,category = make_summary(gby=['day','cat_id'],columns=['cat_id','sum','mean','std'])
fig = px.line(category.reset_index(), x="day", y="sum", color="cat_id",

              line_group="cat_id", hover_name="cat_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="LinePlot over cat_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()
fig = go.Figure()

for i in category.cat_id.value_counts().index:

    fig.add_trace(go.Histogram(x=category.loc[category.cat_id==i,'sum'],name=i))

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Histogram over cat_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
fig = px.box(category, x="cat_id", y="sum",color="cat_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Boxplot over cat_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()
item_sum,item_mean,item_std,item = make_summary(gby=['day','item_id'],columns=['item_id','sum','mean','std'])
fig = px.line(item.reset_index(), x="day", y="sum", color="item_id",

              line_group="item_id", hover_name="item_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="LinePlot over item_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()
fig = go.Figure()

for i in item.item_id.value_counts().index:

    fig.add_trace(go.Histogram(x=item.loc[item.item_id==i,'sum'],name=i))

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Histogram over cat_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
fig = px.box(category, x="item_id", y="sum",color="item_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Boxplot over item_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()
state_sum,state_mean,state_std,state = make_summary(gby=['day','state_id'],columns=['state_id','sum','mean','std'])
fig = px.line(state.reset_index(), x="day", y="sum", color="state_id",

              line_group="state_id", hover_name="state_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="LinePlot over state_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()
fig = go.Figure()

for i in state.state_id.value_counts().index:

    fig.add_trace(go.Histogram(x=state.loc[item.state_id==i,'sum'],name=i))

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Histogram over cat_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)
fig = px.box(state, x="state_id", y="sum",color="state_id")

fig.update_layout(barmode='overlay')

fig.update_traces(opacity=0.75)

fig.update_layout(

    title_text="Boxplot over state_id",

     title_font_size=30,

    autosize=False,

    margin=dict(l=50,r=50,b=100,t=100,pad=4),

    paper_bgcolor="LightSteelBlue",

)

fig.update_xaxes(automargin=True)

fig.show()