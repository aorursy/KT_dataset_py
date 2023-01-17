import numpy as np

import pandas as pd

import seaborn as sns

from bokeh.io import output_notebook, show, reset_output

from bokeh.plotting import figure

from bokeh.models import HoverTool, ColumnDataSource, CDSView, BooleanFilter, Legend

from bokeh.palettes import Spectral



fig =sns.set(style="darkgrid")



import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#cleaned version from www.kaggle.com/rtatman/cleaning-of-commodities-column/output

df = pd.read_csv('/kaggle/input/cleaning-of-commodities-column/cleaned_energy_data.csv')
df.head(10)
df=df.drop(['Unnamed: 0','commodity_transaction','quantity_footnotes'], axis=1)
df=df.dropna(subset=['transaction_type'])
#df[(df.transaction_type.str.contains("production")) & (df.transaction_type.str.contains("total")) & (df.country_or_area == 'Thailand') & (df.year == 2000)]

df[(df.transaction_type.str.contains("consumption")) & (df.country_or_area == 'Thailand') & (df.year == 2000)]
#View Categorical Entries

cat_col=['country_or_area','unit','category', 'commodity', 'transaction_type']



for i in cat_col:

    cat_count = df[i].unique()

    print(cat_count)
#ASEAN

sea_nations= ['Brunei Darussalam', 'Cambodia', 'Indonesia',"Lao People's Dem. Rep.", 'Malaysia', 'Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Viet Nam']





capacity = ['total net installed capacity of electric power plants, combustible fuels',

 'total net installed capacity of electric power plants, geothermal',

 'total net installed capacity of electric power plants, hydro',

 'total net installed capacity of electric power plants, nuclear',

 'total net installed capacity of electric power plants, solar',

 'total net installed capacity of electric power plants, tide, wave, marine',

 'total net installed capacity of electric power plants, wind']



production = ['total geothermal production',

              'total hydro production',

              'total nuclear production',

              'total solar production',

              'total thermal production',

              'total tide, wave production',

              'total wind production']



#did not use

consumption =['consumption by manufacturing, construction and non-fuel mining industry',

              'consumption by transport',

              'consumption by other',

             'final energy consumption']





#did not use

#biofuels = ['fuelwood', 'bagasse', 'animal_waste', 'black_liquor', 'other_vegetal_material_and_residues', 'charcoal', 'biodiesel', 'biogases', 'biogasoline','other_liquid_biofuels']
#creating new dataframe with only needed information

sea_df=df[df.country_or_area.isin(sea_nations)]

sea_df =sea_df[(sea_df.unit=='Kilowatt-hours, million') & ((sea_df.transaction_type.isin(production))|(sea_df.transaction_type == 'final energy consumption'))]
sea_df.country_or_area
transaction_short = {'total geothermal production': 'geothermal',

              'total hydro production':'hydro',

              'total nuclear production':'nuclear',

              'total solar production':'solar',

              'total thermal production':'thermal',

              'total tide, wave production':'tide, wave',

              'total wind production':'wind',

              'final energy consumption': 'final consumption'}



transaction_shorter = {'total geothermal production': 'other renewables',

              'total hydro production':'hydro',

              'total nuclear production':'nuclear',

              'total solar production':'other renewables',

              'total thermal production':'thermal',

              'total tide, wave production':'other renewables',

              'total wind production':'other renewables',

              'final energy consumption' : 'final consumption'}
#new column with shorter name for transaction_type

sea_df['source expanded'] = sea_df.transaction_type.map(transaction_short)

sea_df['source'] = sea_df.transaction_type.map(transaction_shorter)

sea_df.replace({"Lao People's Dem. Rep.":'Lao PDR', 'Brunei Darussalam': 'Brunei'}, inplace =True)
sea_df.country_or_area.unique()
sea_df.shape
sea_df.source.unique()
df_production = sea_df[sea_df.transaction_type.str.contains('production')]

df_consumption = sea_df[sea_df.transaction_type.str.contains('final energy consumption')]
#sea_df[(sea_df.year == 2005) & (sea_df.country_or_area == 'Philippines')]

#sea_df[(sea_df.transaction_type.str.contains('consumption')) & (sea_df.unit == 'Kilowatt-hours, million')].transaction_type.unique()

#sea_df=sea_df.dropna(subset=['transaction_type'])
#dataframe for each country

BRN = sea_df[sea_df.country_or_area =='Brunei'].sort_values('year')

KHM = sea_df[sea_df.country_or_area == 'Cambodia'].sort_values('year')

IDN = sea_df[sea_df.country_or_area == 'Indonesia'].sort_values('year')

LAO = sea_df[sea_df.country_or_area == "Lao People's Dem. Rep."].sort_values('year')

MYS = sea_df[sea_df.country_or_area == 'Malaysia'].sort_values('year')

MMR = sea_df[sea_df.country_or_area == 'Myanmar'].sort_values('year')

PHL = sea_df[sea_df.country_or_area == 'Philippines'].sort_values('year')

SGP = sea_df[sea_df.country_or_area == 'Singapore'].sort_values('year')

THA = sea_df[sea_df.country_or_area == 'Thailand'].sort_values('year')

VNM = sea_df[sea_df.country_or_area == 'Viet Nam'].sort_values('year')
#df[df.transaction_type.str.contains("consumption by")].transaction_type.unique()

#df[(df.transaction_type.str.contains(" production")) & (df.transaction_type.str.contains("total"))].transaction_type.unique()

#PHL[(PHL.transaction_type=='total net installed capacity of electric power plants, geothermal') | (PHL.transaction_type== "total net installed capacity of electric power plants, combustible fuels")]

#PHL[PHL.transaction_type.isin(e_capacity)]

#sea_df[(sea_df.transaction_type=='total net installed capacity of electric power plants, nuclear')] 
total_by_source = pd.pivot_table( sea_df, values = 'quantity', index=['year'], columns =['source'], aggfunc = ('sum'), fill_value=0)

names = ['thermal', 'hydro', 'other renewables']



for i in names:

    total_by_source[i + '_percent'] = (total_by_source[i])*100/(total_by_source[names].sum(axis=1))



cm = sns.light_palette("blue", as_cmap=True)

total_by_source.style.background_gradient(cmap=cm)
Reverse_Spectral= Spectral.copy()

Reverse_Spectral[3] = ('#fc8d59','#ffffbf','#99d594')
from bokeh.layouts import gridplot



output_notebook()



source = ColumnDataSource(total_by_source)

filter_1990 = [True if year == 1990 else False for year in source.data['year']]

filter_2014 = [True if year == 2014 else False for year in source.data['year']]

view1 = CDSView(source=source, filters = [BooleanFilter(filter_1990)])

view2 = CDSView(source=source, filters = [BooleanFilter(filter_2014)])



percentage_names =['thermal_percent','hydro_percent','other renewables_percent']



tooltips_bar = [

    ('source','$name'),

    ('year', '@year'),

    ('KWh, mill.','@$name{0,0} KWh. M')

]



tooltips_source = [

    ('year', '@year'),

    ('By Source', '@$name{0,0} KWh. M'),

    ('Total','$y{0,0} KWh. M')

]

tooltips_mix = [

    ('year', '@year'),

    ('By Source', '@$name{0.00} %')

]



subplot1 = figure( title = 'SEA Electricity Production by Source 1990-2014',plot_width=800, plot_height=500)

subplot1.varea_stack(names,

                x='year',color = Reverse_Spectral[3], legend_label = names, source = source)



subplot1.vline_stack(['thermal', 'hydro', 'other renewables'],

                x='year',color = 'black', source = source)

subplot1.line(x = 'year', y = 'final consumption',

            line_dash=[4, 4], line_color='gray', line_width=2, legend_label='final_consumption', source = source)



subplot1.legend.location ='top_left'

subplot1.add_tools(HoverTool(tooltips = tooltips_source))



subplot2 = figure( title = 'Share of Electricity Production 1990',plot_width=400, plot_height=500)

subplot2.vbar_stack(names,x='year', width=0.1,line_color='black', color= Reverse_Spectral[3], source =source, view=view1)

subplot2.add_tools(HoverTool(tooltips = tooltips_bar))

subplot2.xaxis.ticker=[1990]



subplot3 = figure( title = 'Share of Electricity Production 2014',plot_width=400, plot_height=500)

subplot3.vbar_stack(names,x='year', width=0.1, line_color='black',color= Reverse_Spectral[3], source =source, view=view2)

subplot3.add_tools(HoverTool(tooltips = tooltips_bar))

subplot3.xaxis.ticker=[2014]



# subplot4 = figure( title = 'SEA Electricity Mix by Source 1990-2014',plot_width=400, plot_height=300)

# subplot4.multi_line(y=['thermal_percent','hydro_percent','other renewables_percent'], x='year', color= Reverse_Spectral[3], source = source)

# subplot4.legend.location ='bottom_left'

# subplot4.add_tools(HoverTool(tooltips = tooltips_mix))



grid = gridplot([subplot2, subplot3], ncols=2)

show(subplot1)

show(grid)
total_by_country = pd.pivot_table(df_production, values = 'quantity', index=['year'], columns =['country_or_area'], aggfunc = ('sum'), fill_value=0).sort_values(by= 2014, axis =1, ascending = False)



sea_nations = ['Indonesia', 'Thailand', 'Malaysia', 'Viet Nam', 'Philippines',

       'Singapore', 'Myanmar', "Lao PDR", 'Brunei',

       'Cambodia']



for i in sea_nations:

    total_by_country[i + '_percent'] = round(((total_by_country[i])*100/(total_by_country[sea_nations].sum(axis=1))), 2)



cm = sns.light_palette("blue", as_cmap=True)

total_by_country.style.background_gradient(cmap=cm)
source = ColumnDataSource(total_by_country)

filter_1990 = [True if year == 1990 else False for year in source.data['year']]

filter_2014 = [True if year == 2014 else False for year in source.data['year']]

view1 = CDSView(source=source, filters = [BooleanFilter(filter_1990)])

view2 = CDSView(source=source, filters = [BooleanFilter(filter_2014)])



tooltips_bar = [

    ('source','$name'),

    ('year', '@year'),

    ('KWh, mill.','@$name{0,0} KWh. M')

]



tooltips_source = [

    ('year', '@year'),

    ('By Source', '@$name{0,0} KWh. M'),

    ('Total','$y{0,0} KWh. M')

]

tooltips_mix = [

    ('year', '@year'),

    ('By Source', '@$name{0.00} %')

]



subplot1 = figure(plot_width=800, plot_height=600)

subplot1.varea_stack(sea_nations,

                x='year',color = Spectral[10],legend_label = sea_nations, source = source)



subplot1.vline_stack(sea_nations,

                x='year',color = 'black', source = source)

subplot1.legend.location ='top_left'

subplot1.add_tools(HoverTool(tooltips = tooltips_source))



subplot2 = figure( title = 'Share of Electricity Production 1990',plot_width=400, plot_height=600)

subplot2.vbar_stack(sea_nations,x='year', width=0.1,line_color='black', color= Reverse_Spectral[10], source =source, view=view1)

subplot2.add_tools(HoverTool(tooltips = tooltips_bar))

subplot2.xaxis.ticker=[1990]



subplot3 = figure( title = 'Share of Electricity Production 2014',plot_width=400, plot_height=600)

subplot3.vbar_stack(sea_nations,x='year', width=0.1, line_color='black',color= Reverse_Spectral[10], source =source, view=view2)

subplot3.add_tools(HoverTool(tooltips = tooltips_bar))

subplot3.xaxis.ticker=[2014]



# subplot4 = figure( title = 'SEA Electricity Mix by Source 1990-2014',plot_width=400, plot_height=300)

# subplot4.multi_line(y=['thermal_percent','hydro_percent','other renewables_percent'], x='year', color= Reverse_Spectral[3], source = source)

# subplot4.legend.location ='bottom_left'

# subplot4.add_tools(HoverTool(tooltips = tooltips_mix))



grid = gridplot([subplot2, subplot3], ncols=2)

show(subplot1)

show(grid)
#EnergyMix

total_by_country.iloc[[0,24], 10:]
country_code = [IDN, THA, MYS, VNM,PHL]

country_str = ['IDN', 'THA', 'MYS', 'VNM','PHL']



for i, j in zip(country_code, country_str):

    pivot = pd.pivot_table(i, values = 'quantity', index=['year'], columns =['source'], aggfunc = ('sum'), fill_value=0)

    source = ColumnDataSource(pivot)

    filter_1990 = [True if year == 1990 else False for year in source.data['year']]

    filter_2014 = [True if year == 2014 else False for year in source.data['year']]

    view1 = CDSView(source=source, filters = [BooleanFilter(filter_1990)])

    view2 = CDSView(source=source, filters = [BooleanFilter(filter_2014)])

    

    subplot2 = figure( title = f'{j} Electricity Mix 1990',plot_width=400, plot_height=600)

    subplot2.vbar_stack(names,x='year', width=0.1,line_color='black', color= Reverse_Spectral[3], source =source, view=view1)

    subplot2.add_tools(HoverTool(tooltips = tooltips_bar))

    subplot2.xaxis.ticker=[1990]



    subplot3 = figure(title = f'{j} Electricity Mix 2014',plot_width=400, plot_height=600)

    subplot3.vbar_stack(names,x='year', width=0.1, line_color='black',color= Reverse_Spectral[3], source =source, view=view2)

    subplot3.add_tools(HoverTool(tooltips = tooltips_bar))

    subplot3.xaxis.ticker=[2014]

    grid = gridplot([subplot2, subplot3], ncols=2)

    show(grid)
energylist=['final consumption','hydro','other renewables','thermal']



year = 1990

n = 2014 - year

iloc_index = 24-n



rate_1990  = ((total_by_source[energylist].iloc[-1]/total_by_source[energylist].iloc[iloc_index])**(1/n))-1
year = 2005

n = 2014 - year

iloc_index = 24-n



rate_2005  = ((total_by_source[energylist].iloc[-1]/total_by_source[energylist].iloc[iloc_index])**(1/n))-1
# following trend from 1990

yr_2025 = total_by_source[energylist].iloc[-1]*((1+ rate_1990)**11)

yr_2025.name = 2025



projection1990 = total_by_source.append(yr_2025)



for i in names:

    projection1990[i + '_percent'] = (projection1990[i])*100/(projection1990[names].sum(axis=1))
# following trend from 2005

yr_2025 = total_by_source[energylist].iloc[-1]*((1+ rate_2005)**11)

yr_2025.name = 2025



projection2005 = total_by_source.append(yr_2025)



for i in names:

    projection2005[i + '_percent'] = (projection2005[i])*100/(projection2005[names].sum(axis=1))
growth_rates  = rate_1990.to_frame(name='1990 trend')

growth_rates['2005 trend'] = rate_2005

growth_rates
projection1990
projection1990.iloc[[-1]]

projection2005.iloc[[-1]]
from bokeh.models import BoxAnnotation

source1 = ColumnDataSource(projection1990)

source2 = ColumnDataSource(projection2005)



filter_14_up = [True if [year >= 2014] else False for year in source.data['year']]

filter_14_down = [True if [year < 2014] else False for year in source.data['year']]

view1 = CDSView(source=source, filters = [BooleanFilter(filter_14_up)])

view2 = CDSView(source=source, filters = [BooleanFilter(filter_14_down)])



percentage_names =['thermal_percent','hydro_percent','other renewables_percent']



tooltips_source = [

    ('year', '@year'),

    ('By Source', '@$name{0,0} KWh. M'),

    ('Total','$y{0,0} KWh. M')

]



subplot1 = figure( title = 'SEA Electricity Projection 1',plot_width=800, plot_height=500)

green_box = BoxAnnotation(left=2014, right=22025, fill_color='blue', fill_alpha=0.05)

subplot1.add_layout(green_box)

subplot1.vline_stack(stackers = names, x='year', line_width=4, color = Reverse_Spectral[3],legend_label=names, source = source1)

subplot1.line(x = 'year', y = 'final consumption',

            line_dash=[4, 4], line_color='gray', line_width=2, legend_label='final_consumption', source = source1)



subplot1.legend.location ='top_left'

subplot1.add_tools(HoverTool(tooltips = tooltips_source))





subplot2 = figure( title = 'SEA Electricity Projection 2',plot_width=800, plot_height=500)

subplot2.add_layout(green_box)



subplot2.vline_stack(stackers = names, x='year', line_width=4, color = Reverse_Spectral[3],legend_label=names, source = source2)

subplot2.line(x = 'year', y = 'final consumption',

            line_dash=[4, 4], line_color='gray', line_width=2, legend_label='final_consumption', source = source2)



subplot2.legend.location ='top_left'

subplot2.add_tools(HoverTool(tooltips = tooltips_source))



show(subplot1)

show(subplot2)