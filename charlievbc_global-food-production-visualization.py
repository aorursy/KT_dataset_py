import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import copy

import os



from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, HoverTool, Select, Slider, CustomJS

from bokeh.layouts import column

from bokeh.io import output_notebook, show



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path_var = os.path.join(dirname, filename)



init_df = pd.read_csv(path_var, encoding='latin1')
col_list = list(init_df.columns)

col_remove = ['Area Abbreviation','Area Code','Item Code','Element Code','latitude','longitude']

for i in col_remove:

    col_list.remove(i)



rename_dict = dict()

for i in range(1961,2014):

    rename_dict.update({'Y'+str(i):i})

    

df = init_df[col_list]

df.rename(columns=rename_dict,inplace=True)



# CLEANED DATAFRAME

df
f_total_df = pd.DataFrame(columns=['year','type','total'])

ind=0

for i in ['Food','Feed']:

    f_total = df[df.Element == i].sum(axis=0)

    f_total.drop(f_total.head(4).index,inplace=True)

    f_total.to_dict()

    for k,v in f_total.items():

        f_total_df.loc[ind] = np.nan

        f_total_df['year'][ind] = k

        f_total_df['type'][ind] = i

        f_total_df['total'][ind] = v

        ind +=1

f_total_df['total'] = f_total_df['total'].astype('float')



# RESHAPED DATAFRAME FOR FIRST GRAPH

f_total_df.head()
fig = plt.gcf()

fig.set_size_inches(10,5)



sns.set_style('whitegrid')

sns.lineplot(x='year',y='total',hue='type',data=f_total_df)



plt.xticks(rotation=90)

plt.show()
# DISPLAY BOKEH PLOTS INLINE

output_notebook()
ctry_list = list(df.Area.unique())

yr_list = list(df.columns)

col_remove = ['Area','Item','Element','Unit']

for i in col_remove:

    yr_list.remove(i)



temp_df = pd.DataFrame(columns=yr_list,index=['country','food','feed'])

for yr in yr_list:

    ctry_var = []; food_var = []; feed_var = []

    for ctry in ctry_list:

        ctry_var.append(ctry)

        for type in ['Food','Feed']:

            temp=df[['Area','Element',yr]][(df['Area']==ctry)&(df['Element']==type)]          

            if type == 'Food':

                food_var.append(temp[yr].sum())

            else:

                feed_var.append(temp[yr].sum())

    temp_df[yr]['country'] = ctry_var

    temp_df[yr]['food'] = food_var

    temp_df[yr]['feed'] = feed_var

    

# RESHAPED DATAFRAME FOR SECOND GRAPH

temp_df.head()
temp_dict = temp_df.to_dict()

source = ColumnDataSource(temp_dict[2013])



# DEFINE AND CREATE THE PLOT

plot = figure(title='Food vs Feed per Country',plot_height=300,x_axis_label='Food',y_axis_label='Feed')

plot.circle(x='food',y='feed',source=source,alpha=0.5,size=15)



# CREATE HOVER ITEMS

hover = HoverTool(tooltips=[('Country','@country'),('Food Total','@food'),('Feed Total','@feed')])

plot.add_tools(hover)



# CREATE SLIDER ITEM

slider = Slider(start=1961,end=2013,step=1,value=2013,title='Year (1961-2013)')



# CREATE JAVASCRIPT CALLBACK

callback = CustomJS(args={'source':source,'temp_dict':temp_dict}, code="""

            console.log('changed selected option', cb_obj.value);

            var new_data = temp_dict[cb_obj.value]

            source.data = new_data

            source.change.emit();

""")

slider.js_on_change('value',callback)





# SHOW LAYOUT AND PLOT

show(column(plot,slider))
item_list = list(df.Item.unique())

temp_df = pd.DataFrame(columns=yr_list,index=ctry_list)

for ctry in ctry_list:

    for yr in yr_list:

        item_var = []; count_var = []; element_var = []

        for i in item_list:

            temp = df[['Area','Item','Element',yr]][(df['Area']==ctry)&(df['Item']==i)]

            item_var.append(i)

            count_var.append(temp[yr].sum())

            element_var.append(temp)

        temp_complete = {

            'item' : item_var,

            'count': count_var

        }

        temp_complete_df = pd.DataFrame(temp_complete).sort_values(by='count',ascending=False).head(20)

        temp_df[yr][ctry] = {

            'item' : list(temp_complete_df['item']),

            'count': list(temp_complete_df['count'])

        }

        

# RESHAPED DATAFRAME FOR 3RD GRAPH        

temp_df.head()
temp_dict = temp_df.to_dict()

source = ColumnDataSource(temp_dict[2013]['Philippines'])



# DEFINE AND CREATE PLOT

sorted_cat = sorted(source.data['item'],key=lambda y:source.data['count'][source.data['item'].index(y)],reverse=True)

plot = figure(title='Top 20 Food Items per Country/ per year',plot_height=400,y_range=sorted_cat)

plot.hbar(y='item',right='count',height=0.8,source=source)



# CREATE HOVER ITEMS

hover = HoverTool(tooltips=[('Item','@item'),('Total','@count')])

plot.add_tools(hover)



# CREATE SLIDER AND SELECT ITEMS

slider = Slider(start=1961,end=2013,step=1,value=2013,title='Year (1961-2013)')

select = Select(title='Country',value='Philippines',options=ctry_list)



# CREATE JAVASCRIPT CALLBACK

callback = CustomJS(args={'source':source,'temp_dict':temp_dict,'plot':plot,

                          'slider':slider,'select':select}, code="""

           var new_data = temp_dict[slider.value][select.value]

           plot.y_range.factors = new_data['item']

           plot.change.emit();

           source.data = new_data

           source.change.emit();

""")

slider.js_on_change('value',callback)

select.js_on_change('value',callback)



# SHOW LAYOUT AND PLOT

show(column(select,plot,slider))