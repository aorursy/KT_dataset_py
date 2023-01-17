import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, HoverTool, Select, CustomJS

from bokeh.layouts import column

from bokeh.io import output_notebook, show



import os



# LOAD THE DATA

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = str(os.path.join(dirname, filename))

        if 'inflow' in path:

            inflow = pd.read_csv(path,index_col=0).replace(to_replace=np.nan,value=0)

        elif 'outflow' in path:

            outflow = pd.read_csv(path,index_col=0).replace(to_replace=np.nan,value=0)

        else: 

            bilateral = pd.read_csv(path,index_col=0)
# SAMPLE SNIPPET OF 'INFLOW' DATAFRAME

inflow
# REMOVE UNECESSARY ROWS; ADD A NEW COLUMN CONTAINING TOTAL SUM OF USD PER COUNTRY SINCE 1970s

inflow.drop(inflow.tail(8).index,inplace=True)

inflow['total_in_usd'] = inflow.sum(axis=1)



outflow.drop(outflow.tail(8).index,inplace=True)

outflow['total_in_usd'] = outflow.sum(axis=1)
# SORT VALUES BY HIGHEST USD SUM TOTAL. GET FIRST 20 ONLY.

inflow_top_alltime = inflow.sort_values(by='total_in_usd',ascending=False).head(20)

outflow_top_alltime = outflow.sort_values(by='total_in_usd',ascending=False).head(20)
# RENAME COLUMN AND RESET INDEX

inflow_top_alltime.rename(columns={'Migrant remittance inflows (US$ million)':'Country'}, inplace=True)

inflow_top_alltime.reset_index(drop=True, inplace=True)



outflow_top_alltime.rename(columns={'Migrant remittance outflows (US$ million)':'Country'}, inplace=True)

outflow_top_alltime.reset_index(drop=True, inplace=True)
# SAMPLE CLEANED DATA

inflow_top_alltime.head()
plot = sns.barplot(x='Country',y='total_in_usd',data=inflow_top_alltime)

plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

plt.title('Top 20 countries with most inflows since 1970s (in US$ million)')

plt.show()
plot = sns.barplot(x='Country',y='total_in_usd',data=outflow_top_alltime)

plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

plt.title('Top 20 countries with most outflows since 1970s (in US$ million)')

plt.show()
# LOAD BOKEH PLOTS INLINE

output_notebook()
# TRANSPOSE INFLOW DATAFRAME

transpose_inflow = inflow.set_index('Migrant remittance inflows (US$ million)').transpose()

transpose_inflow.drop(transpose_inflow.tail(2).index,inplace=True) # remove total and 2017p



# TRANSPOSE OUTFLOW DATAFRAME

transpose_outflow = outflow.set_index('Migrant remittance outflows (US$ million)').transpose()

transpose_outflow.drop(transpose_outflow.tail(1).index,inplace=True)



# DISPLAY SAMPLE TRANSPOSED COLUMNS

transpose_outflow.tail()
yr_list = [i for i in range(1970,2017,1)]

col_list = inflow['Migrant remittance inflows (US$ million)'].unique()

temp_df = pd.DataFrame(columns=col_list)

for ctry in col_list:

    temp_df[ctry] = {'year':[], 'inflow':[], 'outflow':[]}

    temp_df[ctry]['year'] = yr_list

    temp_df[ctry]['inflow'] = list(transpose_inflow[ctry])

    temp_df[ctry]['outflow'] = list(transpose_outflow[ctry])



# RESHAPED DATAFRAME FOR CUSTOMJS PURPOSES

temp_df
temp_dict = temp_df.to_dict()

source = ColumnDataSource(temp_dict['Philippines'])



# DEFINE THE GRAPH

plot = figure(title='Inflow vs Outflow per Country (US$ million)',plot_height=400)



# CREATE HOVER ITEMS

hover = HoverTool(tooltips=[('Year','@year'),('Inflow','@inflow{0.2f}'),('Outflow','@outflow{0.2f}')])

plot.add_tools(hover)



# CREATE DROPDOWN SELECTION

select = Select(title='Country List',value='Philippines',options=list(col_list))



# CREATE THE PLOT

plot.diamond(x='year',y='inflow',source=source,color='navy',alpha=0.5,size=15,legend_label='Inflow Total')

plot.line(x='year',y='inflow',source=source,color='navy',line_width=2,line_dash=[4,4])

plot.triangle(x='year',y='outflow',source=source,color='firebrick',alpha=0.5,size=15,legend_label='Outflow Total')

plot.line(x='year',y='outflow',source=source,color='firebrick',line_width=2,line_dash=[4,4])

plot.legend.location = 'top_left'

plot.legend.background_fill_alpha = 0.2



# CREATE CALLBACK

callback = CustomJS(args={'source':source, 'temp_dict':temp_dict}, code="""

           console.log('changed selected option',cb_obj.value);

           var new_data = temp_dict[cb_obj.value]

           source.data = new_data

           source.change.emit();

""")

select.callback = callback



# SHOW LAYOUT AND PLOT 

show(column(select,plot))
bilateral.drop(bilateral.tail(1).index, inplace=True)

bilateral.drop(columns='World', inplace=True)

bilateral.rename(columns={'Remittance-receiving country (across)                                                              -                                                 Remittance-sending country (down) ':'Receiving(across)_Sending(down)'}, inplace=True)

bilateral.set_index('Receiving(across)_Sending(down)', inplace=True)



# CLEANED DATAFRAME

bilateral.head()
temp_df = pd.DataFrame(columns=col_list)

for ctry in col_list:

    temp_df_20 = bilateral[[ctry]].sort_values(by=ctry, ascending=False).head(20)

    temp_df[ctry] = {'country':[], 'total':[]}

    temp_df[ctry]['country'] = list(temp_df_20.index)

    temp_df[ctry]['total'] = list(temp_df_20[ctry])



# RESHAPED DATAFRAME FOR CUSTOMJS PURPOSES

temp_df
temp_dict = temp_df.to_dict()

source = ColumnDataSource(temp_dict['Philippines'])



# DEFINE AND CREATE THE GRAPH

sorted_cat = sorted(source.data['country'], key=lambda y:source.data['total'][source.data['country'].index(y)],reverse=True)

plot = figure(title='Top 20 places where remittances come from (Y2016)',plot_height=400,y_range=sorted_cat)

plot.hbar(y='country',right='total',height=0.8,source=source)



# CREATE HOVER ITEMS

hover = HoverTool(tooltips=[('Total (US$ million)','@total{0.2f}')])

plot.add_tools(hover)



# CREATE DROPDOWN SELECTION

select = Select(title='Country List',value='Philippines',options=list(col_list))



# CREATE CALLBACK

callback = CustomJS(args={'source':source, 'temp_dict':temp_dict, 'plot':plot}, code="""

           console.log('changed selected option',cb_obj.value);

           var new_data = temp_dict[cb_obj.value]

           plot.y_range.factors = new_data['country']

           plot.change.emit();

           source.data = new_data

           source.change.emit();

""")

select.callback = callback



# SHOW LAYOUT AND PLOT 

show(column(select,plot))