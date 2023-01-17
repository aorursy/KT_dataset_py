import pandas as pd
from datetime import datetime
import math
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import DatetimeTickFormatter
df=pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
#create new column: month
year_month=[]
for date in df.date:
    month=datetime.strptime(date, '%Y-%m-%d')
    month=month.strftime('%Y-%m')
    year_month.append(month)
df['month']=year_month
#create new column: year
year=[]
for date in df.date:
    yr=datetime.strptime(date, '%Y-%m-%d')
    yr=yr.year
    year.append(yr)
df['year']=year
#create new column: total number of people affected = n_killed + n_injured
df['n_total']=df.n_killed+df.n_injured
df.head()
df.info()
def create_time_series(category,topn,value):
    #to only show data from 2014 onwards
    data_2014onwards=df[df.year>=2014]
    output_notebook()
    source_all=[]
    
    if category=='all_states': #showing the data for all states
        if value == 'Number of incidents':
            by_month=data_2014onwards[['month','n_total']].groupby(['month']).agg('count').reset_index()
            by_month['category']='all_states'
            max_y=int(math.ceil(by_month['n_total'].max()/10.0))*10
        else:
            by_month=data_2014onwards[['month','n_total']].groupby(['month']).agg('sum').reset_index()
            by_month['category']='all_states'
            max_y=int(math.ceil(by_month['n_total'].max()/10.0))*10
        
        #creating data for bokeh plot
        x_vals = by_month['month'].tolist()
        x = [datetime.strptime(date, "%Y-%m") for date in x_vals]
        y = by_month['n_total'].tolist()
        desc = by_month['category'].tolist()
        #print(by_month)
        data = {'x': x,
                'y': y,
                'desc':desc,
                'month':x_vals,
                'count':y
                }
        source = ColumnDataSource(data)
        source_all.append(source)
    else: #show data based on the category chosen: by state or by city
        if value=='Number of incidents':
            agg='count'
            data_top = data_2014onwards[category].value_counts().to_frame(name='Count').sort_values('Count',ascending=False).head(topn)
            data_top_list = data_top.index.values.tolist()  
        else:
            agg='sum'
            data_top = data_2014onwards[[category,'n_total']].groupby([category]).agg('sum').reset_index()
            data_top = data_top.sort_values('n_total',ascending=False).head(topn)
            data_top_list = list(data_top[category])

        data_top_check=[]
        for data in data_2014onwards[category]:
            if data in data_top_list:
                data_top_check.append(True)
            else: 
                data_top_check.append(False)
        by_category = data_2014onwards[data_top_check][[category,'month','n_total']].groupby([category,'month']).agg(agg)
        by_category = by_category.reset_index()
        by_category.rename(columns={'n_total':agg},inplace=True)
        by_category['Color']='blue'
        by_category['Alpha']=0.1
        max_y=int(math.ceil(by_category[agg].max()/10.0))*10

        for status in data_top_list:
            x_vals = by_category[by_category[category]==status]['month'].tolist()
            x = [datetime.strptime(date, "%Y-%m") for date in x_vals]
            y = by_category[by_category[category]==status][agg].tolist()
            desc = by_category[by_category[category]==status][category].tolist()

            data = {'x': x,
                    'y': y,
                    'desc':desc,
                    'month':x_vals,
                    'count':y
                   }
            source = ColumnDataSource(data)
            source_all.append(source)
    
    plot = figure(plot_width=800, plot_height=400,y_range=(0,max_y*1.1))    
    
    for i in range(0,len(source_all)):
        plot.circle(x='x',y='y', source=source_all[i],fill_color="white",alpha=0.1,name="circle")
        plot.line(x='x',y='y', source=source_all[i],line_width=2,alpha=0.1,hover_line_alpha=1,name="line")
    
    hover = HoverTool(show_arrow=False,
                      line_policy='nearest',
                      names=["line"],
                      tooltips=[('desc','@desc'),
                                ('month','@month'),
                                ('count','@count')]
                     )
    plot.add_tools(hover)
    plot.xaxis.formatter=DatetimeTickFormatter(
        months=["%B %Y"])
    show(plot) 

#creating dropdown widget
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
widgets.Output()

interact(create_time_series, category=['all_states','state','city_or_county'],
         topn = [5,8,10],
         value=['Number of incidents','Total number of people affected']
        )
create_time_series('state',10,'Number of incidents')
create_time_series('city_or_county',10,'Number of incidents')
