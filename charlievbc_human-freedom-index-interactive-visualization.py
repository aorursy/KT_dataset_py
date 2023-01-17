import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import copy



from bokeh.models import ColumnDataSource, HoverTool, Select, Slider, CustomJS, CategoricalColorMapper, Legend

from bokeh.palettes import Turbo11 as palette

from bokeh.layouts import column, row

from bokeh.plotting import figure

from bokeh.io import output_notebook, show



filevar1 = '../input/the-human-freedom-index/hfi_cc_2018.csv'

filevar2 = '../input/the-human-freedom-index/hfi_cc_2019.csv'



try: 

    df_init_2018 = pd.read_csv(filevar1)

    print('File 1 loading - Success!')

    df_init_2019 = pd.read_csv(filevar2)

    print('File 2 loading - Success!')

except:

    print('File loading - Failed!')
# check file difference. use csv with latest records.

print(sorted(df_init_2018.year.unique()))

print(sorted(df_init_2019.year.unique()))
# data information summary

print('Country total:',len(df_init_2019.countries.unique()))

print(df_init_2019.info(verbose=True, null_counts=True))
# a glimpse on the original dataframe

df_init_2019.head()
df = pd.DataFrame()

df[['year','countries','region']] = df_init_2019[['year',

                                                  'countries',

                                                  'region']]

col_int = ['hf_score','hf_rank','pf_rol','pf_ss',

           'pf_ss_women','pf_movement','pf_movement_foreign',

           'pf_movement_women','pf_religion','pf_association',

           'pf_expression','pf_expression_killed',

           'pf_expression_jailed','pf_expression_internet',

           'pf_identity','pf_identity_sex_male',

           'pf_identity_sex_female','pf_identity_divorce',

           'pf_score','pf_rank','ef_government','ef_legal',

           'ef_money','ef_money_inflation','ef_trade',

           'ef_regulation','ef_score','ef_rank']

df[col_int] = df_init_2019[col_int]

df = df.replace(to_replace = "-", value = 0) 

for i in col_int:

    df[i] = df[i].astype(float).astype(int)



# New Dataframe

df
pd.set_option('display.max_rows', None)

table_rank = df[['year','countries','hf_score','hf_rank',

                 'pf_score','pf_rank','ef_score',

                 'ef_rank']][df['year']==2017]

table_rank.sort_values(by='hf_rank')
table_rank.sort_values(by='pf_rank')
table_rank.sort_values(by='ef_rank')
sns.set(font_scale=0.5)

sns.heatmap(data=df.corr(),cmap='BrBG',annot=True,fmt='.2f')

plt.show()
year=[];region=[];hfi_ave=[];

for yr in df.year.unique():

    for rg in sorted(df.region.unique()):

        year.append(yr)

        region.append(rg)

        hfi_ave.append(np.mean(df['hf_score'][(df['year']==yr)&(df['region']==rg)]))

temp_dict = {

    'year':year,

    'region':region,

    'hfi_ave':hfi_ave

}

temp_df = pd.DataFrame(temp_dict)
sns.set(font_scale=1)

sns.set_style('whitegrid')

sns.lineplot(x='year',

             y='hfi_ave',

             hue='region',

             data=temp_df).set(title='Regional HFi Trend',

                               xlabel='Year',

                               ylabel='Average HFi')

plt.legend(bbox_to_anchor=(1.05, 1))

plt.show()
sns.swarmplot(x='hf_score',y='region',data=df[df['year']==2017])

sns.boxplot(x='hf_score',y='region',data=df[df['year']==2017],

            boxprops=dict(alpha=.5)).set(xlabel='HFi',

                                         ylabel='Region',

                                         title='HFi - 2017')

plt.show()
print('TOP COUNTRIES PER REGION')

for i in df.region.unique():

    top_1 = df[['countries','hf_rank']][(df['year']==2017)&(df['region']==i)].sort_values(by='hf_rank').head(1)

    print('>>',i,':',list(top_1['countries'])[0])



print('\nBOTTOM COUNTRIES PER REGION')

for i in df.region.unique():

    bot_1 = df[['countries','hf_rank']][(df['year']==2017)&(df['region']==i)].sort_values(by='hf_rank').tail(1)

    print('>>',i,':',list(bot_1['countries'])[0])    

    
def meltdf(df,idvars,valuevars,varname,valname):

    temp_df = pd.melt(df,id_vars=idvars,

                      value_vars=valuevars,

                      var_name=varname,

                      value_name=valname)

    return temp_df
top_10 = table_rank.sort_values(by='hf_rank').head(10)

bot_10 = table_rank.sort_values(by='hf_rank').tail(10)
melt_bot_10 = meltdf(bot_10,['countries'],['hf_score','pf_score','ef_score'],'indices','scores')

sns.catplot(x='countries',y='scores',hue='indices',data=melt_bot_10,kind='bar').set(title='Bottom 10 HFi - 2017')

plt.xticks(rotation=45)

plt.show()
melt_top_10 = meltdf(top_10,['countries'],['hf_score','pf_score','ef_score'],'indices','scores')

sns.catplot(x='countries',y='scores',hue='indices',data=melt_top_10,kind='bar').set(title='Top 10 Hfi - 2017')

plt.xticks(rotation=45)

plt.show()
# This is to plot graphs inline

output_notebook()
ctry_list = sorted(df.countries.unique())

cols_list = list(df.columns)

cols_list.remove('countries')



df_melt = pd.DataFrame()

for ctry in ctry_list:

    df_melt[ctry] = dict.fromkeys(cols_list, None)

    df_temp = df[df['countries']==ctry]

    for cols in cols_list:

        df_melt[ctry][cols] = list(df_temp[cols])

df_melt.head()
dict_melt = df_melt.to_dict()

source = ColumnDataSource(dict_melt['China'])



# define the plot

plot = figure(title='Freedom Indices',plot_height=300)



# create dropdown selection

select = Select(title='Country',value='China',options=ctry_list)



# create Hover items

cols_plot = copy.copy(cols_list)

for i in ['year','region','hf_rank','pf_rank','ef_rank']:

    cols_plot.remove(i)



hover_list = list()

for i in cols_plot:

    hover_list.append((i, str('@')+i))

hover = HoverTool(tooltips=hover_list)

plot.add_tools(hover)



# create the callback

select.callback = CustomJS(args={'source':source, 

                                 'temp_dict':dict_melt}, 

                           code="""

            console.log('changed selected option',cb_obj.value);

            var new_data = temp_dict[cb_obj.value]

            source.data = new_data

            source.change.emit();

    """)



# create the plot and add legends

lgnd_list = list()

colors = itertools.cycle(palette)

for i in cols_plot:

    a=plot.diamond(x='year',y=i,source=source,alpha=0.5,

                   size=15,color=next(colors))

    b=plot.line(x='year',y=i,source=source,line_width=2,

                line_dash=[4, 4],color=next(colors))

    lgnd_list.append((i,[a,b]))

legend = Legend(items=lgnd_list, location="center")

plot.add_layout(legend, 'right')





show(column(select,plot))
cols_list = list(df.columns)

for i in ['year','hf_rank', 'pf_rank', 'ef_rank']:

    cols_list.remove(i)

cols_list2 = copy.copy(cols_list)

cols_list2.append('x')

cols_list2.append('y')



yr_list = sorted(df.year.unique())

df_melt = pd.DataFrame()

for yr in yr_list:

    df_melt[yr] = dict.fromkeys(cols_list2, None)

    df_temp = df[df['year']==yr]

    for cols in cols_list:

        df_melt[yr][cols] = list(df_temp[cols])

        if cols == 'pf_score':

            df_melt[yr]['x'] = list(df_temp[cols])

        elif cols == 'ef_score':

            df_melt[yr]['y'] = list(df_temp[cols])

        else: 

            continue

df_melt.head()
dict_melt = df_melt.to_dict()

source = ColumnDataSource(dict_melt[2017])



# Define the plot

plot = figure(title='Freedom Indices',plot_width=300,

              plot_height=300)



# Create the Slider and Dropdown Selection

cols_plot = copy.copy(cols_list)

for i in ['countries','region']:

    cols_plot.remove(i)



slider = Slider(start=2008,end=2017,step=1,value=2017,

                title='Year')

select_x = Select(title='X-Axis Selection',

                  value='pf_score',

                  options=cols_plot)

select_y = Select(title='Y-Axis Selection',

                  value='ef_score',

                  options=cols_plot)



# Create the Hover items

hover = HoverTool(tooltips=[('Country','@countries'),

                            ('Region','@region')])

plot.add_tools(hover)



# Create the plot

color_mapper = CategoricalColorMapper(factors=sorted(df.region.unique()),palette=palette)

plot.circle(x='x',y='y',source=source,alpha=0.5,radius=.3,

           color=dict(field='region',transform=color_mapper))



# Create the callback

callback = CustomJS(args={'source':source,'temp_dict':dict_melt,

                         'xselect':select_x,'yselect':select_y,

                         'slider':slider}, code="""

            var new_data = temp_dict[slider.value]

            source.data['x']= new_data[xselect.value]

            source.data['y']= new_data[yselect.value]

            source.data['countries']= new_data['countries']

            source.data['region']= new_data['region']

            source.change.emit();          

    """)



select_x.js_on_change('value', callback)

select_y.js_on_change('value', callback)

slider.js_on_change('value', callback)





show(row(select_y,column(plot,select_x,slider)))