# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Bokeh libraries

from bokeh.io import output_notebook

from bokeh.palettes import inferno, Spectral, all_palettes, Category20

from bokeh.plotting import figure, curdoc, show

from bokeh.models import ColumnDataSource, FactorRange, CustomJS, Slider

from bokeh.models.widgets import Panel, Tabs

from bokeh.layouts import gridplot, row, column, layout



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

import warnings

import numpy as np # we will use this later, so import it now

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from bokeh.io import output_notebook, show

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,LinearInterpolator

from datetime import datetime

from dateutil.parser import parse

import pandas as pd

import collections





from bokeh.io import output_file, show

from bokeh.layouts import row

from bokeh.palettes import Viridis3

from bokeh.plotting import figure

from bokeh.models import CheckboxGroup, CustomJS

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn,Div

import numpy as np

from bokeh.models import Paragraph, Panel, Tabs, Column



from bokeh.io import output_file, show

from bokeh.layouts import row

from bokeh.palettes import Viridis3

from bokeh.plotting import figure

from bokeh.models import CheckboxGroup, CustomJS

from bokeh.layouts import column, widgetbox

from bokeh.models import CustomJS, ColumnDataSource, Slider, Select

from bokeh.plotting import figure, output_file, show

from bokeh.models.glyphs import VBar

from bokeh.palettes import Spectral6

from bokeh.transform import linear_cmap,factor_cmap

# from collections import Counter

from math import pi

from bokeh.models.glyphs import Wedge

from bokeh.models import glyphs,GraphRenderer



import pandas as pd



from bokeh.palettes import Category20c,Spectral

from bokeh.plotting import figure, show

from bokeh.transform import cumsum

import pandas as pd



output_notebook()

# My word count data

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Title: Interactive Sales Dashboard

# Author: Akhilesh Rai

# Date: 31/08/2019

# Thank you for checking out my kernel!  

# Please upvote if you like the dashboard and comment if you have any suggestions.



sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

items_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

year = []

month = []

for x in range(0,sales_train.shape[0]):

    month.append(sales_train['date'][x].split('.')[1])

    year.append(sales_train['date'][x].split('.')[2])

sales_train['date'] = sales_train['date'].str.replace('.', '-', regex=True)

sales_train['month'] = month

sales_train['year'] = year

sales_df = sales_train.copy()

df = sales_df.copy()

source_sales = ColumnDataSource(data=df)

year_list=df['year'].unique().tolist().copy()

month_list = df['month'].unique().tolist().copy()

month_list = list(sales_train.month)

month_count = collections.Counter(month_list)

month = list(month_count.keys())

month_counts = list(month_count.values())

year_list = list(sales_train.year)

year_count = collections.Counter(year_list)

year = list(year_count.keys())

year_count = list(year_count.values())

month_group = pd.DataFrame({'index': list(range(1,len(sales_train.groupby(["month"])["item_cnt_day"].sum())+1)),

                                'month_count': sales_train.groupby(["month"])["item_cnt_day"].sum()})



#ts.drop(columns = 'date_block_num')



month_group = month_group.reset_index()

del month_group['month']

month_group.astype({'month_count': 'int32'}).dtypes

month_cds = ColumnDataSource(month_group)



hover = HoverTool(tooltips = '@month_count= Counts')

sales_month = figure(plot_width=400, plot_height=300,tools = [hover])



sales_month.line('index','month_count',source = month_cds, line_width=2)

sales_month.background_fill_color = '#fffce6'



year_group = pd.DataFrame({'index': list(range(1,len(sales_train.groupby(["year"])["item_cnt_day"].sum())+1)),

                                'year_count': sales_train.groupby(["year"])["item_cnt_day"].sum()})



#ts.drop(columns = 'date_block_num')



year_group = year_group.reset_index()

del year_group['year']

year_group.astype({'year_count': 'int32'}).dtypes

year_cds = ColumnDataSource(year_group)



hover = HoverTool(tooltips = '@year_count= Counts')

sales_year = figure(plot_width=400, plot_height=300,tools = [hover])



sales_year.line('index','year_count',source = year_cds, line_width=2)

sales_year.background_fill_color = '#fffce6'



ts = pd.DataFrame({'index': list(range(1,len(sales_train.groupby(["date_block_num"])["item_cnt_day"].sum())+1)),

                                'datenum': sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()})



#ts.drop(columns = 'date_block_num')

ts.head()

ts = ts.reset_index()

del ts['date_block_num']

ts.astype({'datenum': 'int32'}).dtypes

ts_cds = ColumnDataSource(ts)



hover = HoverTool(tooltips = '@datenum= Counts')

sales_line = figure(plot_width=400, plot_height=300,tools = [hover])



sales_line.line('index','datenum',source = ts_cds, line_width=2)

sales_line.background_fill_color = '#fffce6'



sales_line.axis.visible=False



output_notebook()

hover = HoverTool()

p = figure(plot_width=400, plot_height=400,tools = [hover])

p.background_fill_color = '#fffce6'

p.outline_line_width = 7

p.outline_line_alpha = 0.9

p.outline_line_color = "black"



props = dict(line_width=4, line_alpha=0.7)

l0 = p.line('index','datenum',source = ts_cds, color=Viridis3[0],legend = "Line 0 : per day", **props)

l1 = p.line('index','month_count',source = month_cds, color=Viridis3[2], legend="Line 1: per month", **props)



checkbox = CheckboxGroup(labels=["Line 0", "Line 1"],

                         active=[0, 1], width=100)

checkbox.callback = CustomJS(args=dict(l0=l0, l1=l1, checkbox=checkbox), code="""

l0.visible = 0 in checkbox.active;

l1.visible = 1 in checkbox.active;



""")



layout_sales_line = column(checkbox, p)





Year_COLORS =  ['#2b83ba',

                '#abdda4',

                '#8e0152']

COLORS = ['#2b83ba',

          '#abdda4',

          '#ffffbf',

          '#fdae61',

          '#d7191c',

          '#276419',

          '#4d9221',

          '#7fbc41',

          '#b8e186',

          '#e6f5d0',

          '#c51b7d',

          '#8e0152']

# create dataframes of data





month_counts = collections.Counter(month_list)

df_month = pd.DataFrame({'months' : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul','Aug','Sept','Oct','Nov','Dec'],

                         'counts':list(month_counts.values()),'color' : COLORS})



source_month = ColumnDataSource(df_month)





df_year = pd.DataFrame({'years' : year,

                         'counts':year_count,'color' : Year_COLORS})



source_year = ColumnDataSource(df_year)



total = sum(list(month_counts.values()))

x = [1,2,3,4,5,7,8,9,10,11,12]

y = [1,2,3,4,5,7,8,9,10,11,12]







months = df_month['months']



hover = HoverTool(tooltips = '@counts= Counts')

bar_plot = figure(x_range=months, plot_height=400,plot_width =400, toolbar_location=None, title="No of Items Sold", tools = [hover])

bar_months = bar_plot.vbar(x='months', top='counts', width=0.9, source=source_month,

                   line_color='white',color = 'color')

bar_years = bar_plot.vbar(x='years', top='counts', width=0.9, source=source_year,

                   line_color='white',color = 'color')





bar_plot.background_fill_color = '#fffce6'

bar_plot.outline_line_width = 7

bar_plot.outline_line_alpha = 0.9

bar_plot.outline_line_color = "black"

# callback for input controls

callback = CustomJS(args=dict(months=bar_months, years=bar_years, plot=bar_plot), code="""

    if (ui_view.value=="months") {

      plot.x_range.factors = months.data_source.data.months

    } else {

      plot.x_range.factors = years.data_source.data.years

    }

""")



ui_view = Select(title="View", callback=callback, value="months", options=["months", "years"])

callback.args['ui_view'] = ui_view



# layout

layout_bar = column(ui_view, bar_plot)







df_month = df_month.rename(index=str, columns={0:'value', 'index':'months'})

df_month['angle'] = df_month['counts']/sum(list(df_month['counts'])) * 2*pi

df_month['color'] = Category20c[len(list(df_month['counts']))]



df_year = pd.DataFrame({'index' : [x for x in range(1,4)],'years' : year,

                         'counts': year_count})

df_year = df_year.rename(index=str, columns={0:'value'})



df_year['angle'] = df_year['counts']/sum(list(df_year['counts'])) * 2*pi

df_year['color'] = Spectral[len(list(df_year['counts']))]



source_pie_month = ColumnDataSource(df_month)

source_pie_year = ColumnDataSource(df_year)

source = ColumnDataSource(df_month)





pie_chart = figure(plot_height=400,plot_width = 500, title="Pie Chart", toolbar_location=None,

           tools="hover", tooltips=[("Counts", "@counts")])

pie_chart.wedge(x=0, y=1, radius=0.4, 

                       start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

                       line_color="white", fill_color='color', legend='counts', source=source)



pie_chart.legend.location = "bottom_right"

pie_chart.legend.orientation = "vertical"

pie_chart.legend.label_text_font_size = "8px"

pie_chart.background_fill_color = '#fffce6'

pie_chart.xgrid.visible = False

pie_chart.ygrid.visible = False

pie_chart.axis.axis_label=None

pie_chart.axis.visible=False

pie_chart.grid.grid_line_color = None

pie_chart.outline_line_width = 7

pie_chart.outline_line_alpha = 0.9

pie_chart.outline_line_color = "black"



select = Select(options=['month', 'year'],value='month',title = "Value")

callback = CustomJS(args={"cds2": source_pie_month, "cds3": source_pie_year, "source": source}, code="""

    if (cb_obj.value == "month") {

        source.data = cds2.data

    } else if(cb_obj.value == "year") {

        source.data = cds3.data

    }

    console.log(source.data)

""")

select.js_on_change('value', callback)

layout_pie = column(select, pie_chart)

df_month = df_month.drop(['color','angle'],axis = 1)



Columns = [TableColumn(field=Ci, title=Ci) for Ci in df_month.columns] # bokeh columns

data_table = DataTable(columns=Columns, source=ColumnDataSource(df_month),width = 300) # bokeh table



some_div = Div(text="<b> No of Items sold: </b>", style={'font-size': '200%', 'color': 'black'})

more_div = Div(text="<b> 2935849 </b>", style={'font-size': '200%', 'color': '#5f9afa'})

y = column(row(some_div,more_div),data_table)



l = gridplot([[layout_sales_line,y],[layout_bar,layout_pie]])

title_div = Div(text="<b> Sales Count </b>", style={'font-size': '400%', 'color': '#5f9afa'})

p2 = column(title_div,l)



month_group = pd.DataFrame({'index': list(range(1,len(sales_train.groupby(["month"])["item_price"].sum())+1)),

                                'month_count': sales_train.groupby(["month"])["item_price"].sum()})



#ts.drop(columns = 'date_block_num')

month_group.head()

month_group = month_group.reset_index()

del month_group['month']

month_group.astype({'month_count': 'int32'}).dtypes

month_price_cds = ColumnDataSource(month_group)



hover = HoverTool(tooltips = '@month_count= Counts')

sales_price_month = figure(plot_width=400, plot_height=300,tools = [hover])



sales_price_month.line('index','month_count',source = month_price_cds, line_width=2)

sales_price_month.background_fill_color = '#fffce6'



year_group = pd.DataFrame({'index': list(range(1,len(sales_train.groupby(["year"])["item_price"].sum())+1)),

                                'year_count': sales_train.groupby(["year"])["item_price"].sum()})



#ts.drop(columns = 'date_block_num')



year_group = year_group.reset_index()

del year_group['year']

year_group.astype({'year_count': 'int32'}).dtypes

year_price_cds = ColumnDataSource(year_group)



ts_price = pd.DataFrame({'index': list(range(1,len(sales_train.groupby(["date_block_num"])["item_price"].sum())+1)),

                                'datenum': sales_train.groupby(["date_block_num"])["item_price"].sum()})



#ts.drop(columns = 'date_block_num')

ts_price.head()

ts_price = ts_price.reset_index()

del ts_price['date_block_num']

ts_price.astype({'datenum': 'int32'}).dtypes

ts_price_cds = ColumnDataSource(ts_price)



hover = HoverTool(tooltips = '@datenum= Counts')

sales_line = figure(plot_width=500, plot_height=300,tools = [hover])



sales_line.line('index','datenum',source = ts_price_cds, line_width=2)

sales_line.background_fill_color = '#fffce6'



sales_line.axis.visible=False



output_notebook()

hover = HoverTool()

p = figure(plot_width=400, plot_height=400,tools = [hover])

p.background_fill_color = '#fffce6'

p.outline_line_width = 7

p.outline_line_alpha = 0.9

p.outline_line_color = "black"



props = dict(line_width=4, line_alpha=0.7)

l0 = p.line('index','datenum',source = ts_price_cds, color=Viridis3[0],legend = "Line 0 : per day", **props)

l1 = p.line('index','month_count',source = month_price_cds, color=Viridis3[2], legend="Line 1: per month", **props)



checkbox = CheckboxGroup(labels=["Line 0", "Line 1"],

                         active=[0, 1], width=100)

checkbox.callback = CustomJS(args=dict(l0=l0, l1=l1, checkbox=checkbox), code="""

l0.visible = 0 in checkbox.active;

l1.visible = 1 in checkbox.active;



""")



layout_sales_line2 = column(checkbox, p)



Year_COLORS =  ['#2b83ba',

                '#abdda4',

                '#8e0152']

COLORS = ['#2b83ba',

          '#abdda4',

          '#ffffbf',

          '#fdae61',

          '#d7191c',

          '#276419',

          '#4d9221',

          '#7fbc41',

          '#b8e186',

          '#e6f5d0',

          '#c51b7d',

          '#8e0152']

# create dataframes of data





df_month2 = pd.DataFrame({'months' : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul','Aug','Sept','Oct','Nov','Dec'],

                         'counts':list(month_group['month_count']),'color' : COLORS})



source_month2 = ColumnDataSource(df_month2)





df_year2 = pd.DataFrame({'years' : year,

                         'counts':list(year_group['year_count']),'color' : Year_COLORS})



source_year2 = ColumnDataSource(df_year2)



total = sum(list(month_group['month_count']))

x = [1,2,3,4,5,7,8,9,10,11,12]

y = [1,2,3,4,5,7,8,9,10,11,12]







months = df_month['months']



hover = HoverTool(tooltips = '@counts= Price')

bar_plot2 = figure(x_range=months, plot_height=400,plot_width =400, toolbar_location=None, title="Total Sales", tools = [hover])

bar_months = bar_plot2.vbar(x='months', top='counts', width=0.9, source=source_month2,

                   line_color='white',color = 'color')

bar_years = bar_plot2.vbar(x='years', top='counts', width=0.9, source=source_year2,

                   line_color='white',color = 'color')





bar_plot2.background_fill_color = '#fffce6'

bar_plot2.outline_line_width = 7

bar_plot2.outline_line_alpha = 0.9

bar_plot2.outline_line_color = "black"

# callback for input controls

callback = CustomJS(args=dict(months=bar_months, years=bar_years, plot=bar_plot2), code="""

    if (ui_view.value=="months") {

      plot.x_range.factors = months.data_source.data.months

    } else {

      plot.x_range.factors = years.data_source.data.years

    }

""")



ui_view = Select(title="View", callback=callback, value="months", options=["months", "years"])

callback.args['ui_view'] = ui_view



# layout

layout_bar2 = column(ui_view, bar_plot2)



df_month2 = df_month2.rename(index=str, columns={0:'value', 'index':'months'})

df_month2['angle'] = df_month2['counts']/sum(list(df_month2['counts'])) * 2*pi

df_month2['color'] = Category20c[len(list(df_month2['counts']))]





df_year2 = df_year2.rename(index=str, columns={0:'value'})



df_year2['angle'] = df_year2['counts']/sum(list(df_year2['counts'])) * 2*pi

df_year2['color'] = Spectral[len(list(df_year2['counts']))]



source_pie_month2 = ColumnDataSource(df_month2)

source_pie_year2 = ColumnDataSource(df_year2)

source = ColumnDataSource(df_month2)



pie_chart2 = figure(plot_height=400,plot_width = 500, title="Pie Chart", toolbar_location=None,

           tools="hover", tooltips=[("Counts", "@counts")])

pie_chart2.wedge(x=0, y=1, radius=0.4, 

                       start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

                       line_color="white", fill_color='color', legend='counts', source=source)



pie_chart2.legend.location = "bottom_right"

pie_chart2.legend.orientation = "vertical"

pie_chart2.legend.label_text_font_size = "8px"

pie_chart2.background_fill_color = '#fffce6'

pie_chart2.xgrid.visible = False

pie_chart2.ygrid.visible = False

pie_chart2.axis.axis_label=None

pie_chart2.axis.visible=False

pie_chart2.grid.grid_line_color = None

pie_chart2.outline_line_width = 7

pie_chart2.outline_line_alpha = 0.9

pie_chart2.outline_line_color = "black"



select = Select(options=['month', 'year'],value='month',title = "Value")

callback = CustomJS(args={"cds2": source_pie_month2, "cds3": source_pie_year2, "source": source}, code="""

    if (cb_obj.value == "month") {

        source.data = cds2.data

    } else if(cb_obj.value == "year") {

        source.data = cds3.data

    }

    console.log(source.data)

""")

select.js_on_change('value', callback)

layout_pie2 = column(select, pie_chart2)



df_month2 = df_month2.drop(['color','angle'],axis = 1)



Columns = [TableColumn(field=Ci, title=Ci) for Ci in df_month2.columns] # bokeh columns

data_table2 = DataTable(columns=Columns, source=ColumnDataSource(df_month2),width = 300) # bokeh table



some_div = Div(text="<b> Total Sales: </b>", style={'font-size': '200%', 'color': 'black'})

more_div = Div(text="<b> 2615410572.36 </b>", style={'font-size': '200%', 'color': '#5f9afa'})

y2 = column(row(some_div,more_div),data_table2)



l2 = gridplot([[layout_sales_line2,y2],[layout_bar2,layout_pie2]])

title_div2 = Div(text="<b> Total Sales </b>", style={'font-size': '400%', 'color': '#5f9afa'})

p = column(title_div2,l2)











#more_div = Div(text="<b> 2615410572.36 </b>", style={'font-size': '200%', 'color': '#000000'})

title_div2 = Div(text="<b> About </b>", style={'font-size': '400%', 'color': '#5f9afa'})

div = Div(text=""" The kernel was made after much delibration with folks at <a href="http://discourse.bokeh.org">bokeh.org</a>. Remember that this is just part 1 of the notebook and

the prediction part is due in the <b>next</b> notebook. 

I would say that this part <b>EDA</b> is yet to be completed. 

<p>The major concept of Bokeh is that graphs are built up one layer at a time. We start out by creating a figure, and then we add elements, called glyphs, to the figure. 

                        (For those who have used ggplot, the idea of glyphs is essentially the same as that of geoms which are added to a graph one ‘layer’ at a time.) 

                            Glyphs can take on many shapes depending on the desired use: circles, lines, patches, bars, arcs, and so on.

                            </p><h2>References:</h2> 

  

<ul> 

  <li><a href="http://discourse.bokeh.org">Bokeh.org</a></li> 

  <li><a href="https://bokeh.pydata.org/en/latest"/>Bokeh Documentation</li> 

  <li><a href="<a href="https://stackoverflow.com/questions/45197006/how-to-make-an-interactive-bokeh-plot"/>Stack Overflow</li>  

</ul>  </p>

<p>&copy Akhilesh Rai.<i>2019</i>. All Rights Reserved.</p>

""", width=700, height=100)



#last_div = Div(text="<p></p><p></p>&copy Akhilesh Rai", style={'font-size': '80%', 'color': '#5f9afa'})

titlediv = column(div)



text = column(title_div2,titlediv)

p3 = column(text)





tab1 = Panel(child=p,title="Total Sales")

tab2 = Panel(child=p2, title="Sales Count")

tab3 = Panel(child=p3, title="About")

tabs = Tabs(tabs=[ tab1, tab2,tab3 ])

show(tabs)