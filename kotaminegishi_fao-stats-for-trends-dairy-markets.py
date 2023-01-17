import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py 

import plotly.graph_objs as go
# The data for 1961-2013

df0 = pd.read_csv("../input/world-foodfeed-production/FAO.csv",  

                  encoding = "ISO-8859-1")

df0.head()
df0.Element.value_counts()
# Use the whole data from FAO STAT for 1961-2013  

df1 = pd.read_csv("../input/fao-foodbalancesheets/FoodBalanceSheetsHistoric_E_All_Data.csv",

                  encoding = "ISO-8859-1")

df1.head()
# Stack data over Year

df1_columns = df1.columns

df1_yrs = df1_columns[~(df1_columns.str.endswith("F")) &

                      df1_columns.str.startswith("Y")]



df1_stack = df1.melt(id_vars = df1_columns[0:7], 

               var_name = "Year",

               value_vars=df1_yrs, value_name="Value")

df1_stack['Year'] = df1_stack.Year.str.slice(start=1).astype(int)

df1_stack.head()
# The data for 2014-2017 and stack it over Year

df2 = pd.read_csv("../input/fao-foodbalancesheets/FoodBalanceSheets_E_All_Data.csv",  encoding = "ISO-8859-1")

df2_columns = df2.columns

df2_yrs = df2_columns[~(df2_columns.str.endswith("F")) & 

                      df2_columns.str.startswith("Y")]



df2_stack = df2.melt(id_vars = df2_columns[0:7], 

               var_name = "Year",

               value_vars=df2_yrs, value_name="Value")

df2_stack['Year'] = df2_stack.Year.str.slice(start=1).astype(int)

df2_stack.head()
# Combine df1 and df2 by stacking

df = df1_stack.append(df2_stack, ignore_index=True)

df.head()
df.shape
# Check what's in "Area Code" 

df['Area Code'].unique()
df[df['Area Code'] >= 5000].Area.unique()
df[df['Area Code'] < 5000].Area.unique()
# Check what's in "Item" 

df.Item.value_counts()[:50] # show first 50
# Check what's in "Element" 

df.Element.value_counts()
US_2017_milk = df[(df.Area == 'United States of America') 

            & (df.Year == 2017)

            & (df.Item == 'Milk - Excluding Butter' )

            & (df['Item Code']==2948) ]



US_2017_milk 
df[(df.Area == 'United States of America') 

  & (df.Year == 2017)

   & (df.Element == 'Total Population - Both sexes' )]
254.87 * 325084.76 / 1000
97787 - 11346 + 1087 - 946 
df_prod = df[df.Element == 'Production']

df_cons = df[df.Element == 'Domestic supply quantity']

df_cons_cap = df[df.Element == 'Food supply quantity (kg/capita/yr)']

df_pop = df[df.Element == 'Total Population - Both sexes']

df_import = df[df.Element == 'Import Quantity']

df_export = df[df.Element == 'Export Quantity']
# Define US production subset data

US_code = (df_prod[df_prod.Area=='United States of America']

           ['Area Code'].value_counts().index)

df_prod_US = df_prod[(df_prod['Area Code'] == US_code[0])] 
# US Milk production

US_milk = df_prod_US[(df_prod_US.Item == 'Milk - Excluding Butter') &

          (df_prod_US['Item Code'] == 2948)]

US_milk.head()
# US Corn production

US_corn = df_prod_US[df_prod_US.Item.str.contains('Maize and products',

                                                  case=False)]

US_corn.head()
# US Soyvean production

US_soybean = df_prod_US[df_prod_US.Item.str.contains('Soyabeans',

                                                     case=False)]

US_soybean.head()
# Observe what kind of meat products are in the data

df_prod_US[df_prod_US.Item.str.contains('meat',case=False)]
# US Beef production

US_beef = df_prod_US[df_prod_US.Item.str.contains('Bovine',

                                                  case=False)]

US_beef.head()
# US Poultry production

US_poultry = df_prod_US[df_prod_US.Item.str.contains('Poultry',

                                                     case=False)]

US_poultry.head()
# US Pork production

US_pork = df_prod_US[df_prod_US.Item.str.contains('Pigmeat',

                                                  case=False)]

US_pork.head()
World_milk_total = df_prod[

    (df_prod.Item == 'Milk - Excluding Butter') &

    (df_prod['Item Code'] == 2948) & 

    (df_prod.Area == "World")]



World_milk_total['Item'] = 'Milk, Total World Production'

World_milk_total.head()
# Define US milk and other comparison dataset  

US_prod = (US_milk

           .append(US_soybean)

           .append(US_corn)

           .append(US_beef)

           .append(US_pork)

           .append(US_poultry)

           .append(World_milk_total)

          )
# Normalize at the value of 1980 as the baseline 

US_prod = US_prod.assign(Value_1980 = (US_prod.Year==1980) * US_prod.Value)

US_prod['Value_1980'] = (US_prod.groupby('Item').Value_1980

                    ).transform(max)

US_prod['Value_base80'] = US_prod.Value/US_prod.Value_1980



US_prod[100:150]
sns.relplot(

    x = 'Year', y = 'Value_base80', hue = 'Item', kind = 'line',

    data = US_prod.query('Year >= 1980'),

    aspect = 1.5

);
# see https://plot.ly/python/line-charts/



import plotly.graph_objects as go

fct_resize = .75



y_vars = ['Milk - Excluding Butter',

         'Bovine Meat',

         'Pigmeat',

         'Maize and products',

         'Milk, Total World Production']

labels = ['US Dairy', 'US Beef', 'US Pork', 'US Corn', 'World Dairy']

colors = ['#3498db', ' #e74c3c ', '#9b59b6', ' #f1c40f', ' #16a085']



mode_size = [12, 8, 8, 8, 12]

line_size = [5, 3, 3, 3, 5]



x_data = np.arange(1980, 2018)



y_data = np.empty(shape = [len(y_vars), len(x_data)])



for i in range(0, 5):

    y_data[i] = US_prod[(US_prod.Item == y_vars[i]) & 

           (US_prod.Year >= 1980)].Value_base80





 

fig = go.Figure()



for i in range(0, 5):

    fig.add_trace(go.Scatter(x=x_data, y=y_data[i], mode='lines',

        name=labels[i],

        line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,

    ))



    # endpoints

    fig.add_trace(go.Scatter(

        x=[x_data[-1]],

        y=[y_data[i][-1]],

        mode='markers',

        marker=dict(color=colors[i], size=mode_size[i])

    ))



    

fig.update_layout(

    height= 600 * fct_resize,

    width = 800 * fct_resize,

    title = 'Fig 1. Growth of Selected Commodity Production, 1980-2017',

    yaxis_title="Production, base=1980",

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2 * fct_resize,

        ticks='outside',

        tickfont=dict(

            #family='Arial',

            size=15 * fct_resize,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=True,

        showline=True,

        linecolor='rgb(204, 204, 204)',

        zerolinecolor='#D3D3D3',

        showticklabels=True,

        ticks='outside',

        tickfont=dict(

            #family='Arial',

            size=15 * fct_resize,

            color='rgb(82, 82, 82)',

        ),

    ),

    font=dict(size=15 * fct_resize),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=60,

        r=150,

        t=40,

    ),

    showlegend=False,

    plot_bgcolor='white',

)



annotations = []



annotations.append(

    dict(xref='paper', yref='paper',

        x= 1, #x= 0.02,

        xanchor= 'right',

        y= 1, #y=-.12,

        yanchor= 'bottom',

        showarrow=False,

        text= 'Data source: FAOSTAT.'))



for y_trace, label, color in zip(y_data, labels, colors):

    

    # labeling the right_side of the plot

    annotations.append(

        dict(xref='paper', x=0.95, y=y_trace[-1],

              xanchor='left', yanchor='middle',

              text='+' '{}%'.format(int(round(y_trace[-1]*100)-100)) + ' ' + label,

              font=dict(#family='Arial',

                        size=16 * fct_resize),

              showarrow=False))

        

fig.update_layout(annotations=annotations)    

fig.show()
(df_import[(df_import.Item == 'Milk - Excluding Butter') &

                         (df_import['Item Code'] == 2948) &

                         (df_import['Area Code'] < 500) &

                         (df_import.Year >= 2015)])
import_milk = (df_import[(df_import.Item == 'Milk - Excluding Butter') &

                         (df_import['Item Code'] == 2948) &

                         (df_import['Area Code'] < 500) &

                         (df_import.Year >= 2015)]

              [['Area', 'Value']]

               .groupby(['Area']).agg('mean')

              ).sort_values('Value', ascending = False).reset_index()

import_milk.Area = import_milk.Area.replace({'Russian Federation': 'Russia', 

                                            'United States of America': 'United States'})

import_milk = import_milk.set_index('Area')

import_milk.head(10).round()
export_milk = (df_export[(df_export.Item == 'Milk - Excluding Butter') &

                         (df_export['Item Code'] == 2948) &

                         (df_export['Area Code'] < 500) &

                         (df_export.Year >= 2015)]

              [['Area', 'Value']]

               .groupby(['Area']).agg('mean')

              ).sort_values('Value', ascending = False).reset_index()

export_milk.Area = export_milk.Area.replace({'Russian Federation': 'Russia', 

                                            'United States of America': 'United States'})

export_milk = export_milk.set_index('Area')

export_milk.head(10).round()
dairy_exporters = (export_milk.iloc[range(0,11)]

                    .append(export_milk.loc[import_milk[:10].index])

                    .drop_duplicates().rename(columns={'Value':'Export'})

                    .sort_values(by = 'Export', ascending=True)

                    )



dairy_exporters
dairy_importers = (import_milk.iloc[range(0,11)]

                    .append(import_milk.loc[export_milk[:10].index])

                    .drop_duplicates().rename(columns={'Value':'Import'})

                    .reindex(dairy_exporters.index)

                  ).fillna(0)

dairy_importers 
import plotly.graph_objects as go



fct_resize = .75 



fig = go.Figure()

fig.add_trace(go.Scatter(

    x=dairy_exporters.Export,

    y=dairy_exporters.index,

    marker=dict(color="#2593ff", size=15 * fct_resize),

    mode="markers",

    name="Export",

))



fig.add_trace(go.Scatter(

    x=dairy_importers.Import,

    y=dairy_importers.index,

    marker=dict(color="#ff8e25", size=15 * fct_resize),

    mode="markers",

    name="Import",

))



fig.update_layout(title="Fig 2. Major Dairy Exporters and Importers, 2015-17 Average",

                  xaxis_title="1,000 Metric tons",

                  yaxis_title="",

                    width=800 * fct_resize,

                    height=600 * fct_resize,

                    margin=dict(l=40, r=40, b=40, t=40),

                    font=dict(size=15 * fct_resize),

                    legend=dict(

                        font_size=14 * fct_resize,

                        x = 1,

                        y = 0.1,

                        yanchor='bottom',

                        xanchor='right',

                    ),

                   #paper_bgcolor='white',

                     plot_bgcolor='white',

                      annotations=[go.layout.Annotation(

                        xref='paper',

                        yref='paper',

                        x= 1, #x= 0.02,

                        xanchor= 'right',

                        y= 1, #y=-.12,

                        yanchor= 'bottom',

                        showarrow=False,

                        text= 'Data source: FAOSTAT.'),]

                 )

fig.update_xaxes(showline=False, #linewidth=2, linecolor= '#D3D3D3', 

                 showgrid=True, gridwidth=1, gridcolor= '#D3D3D3',

                 zeroline=True, zerolinewidth=2, zerolinecolor='#D3D3D3')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor= 'slategray')



fig.show()

# Recall regional aggregate area names

df_pop[df_pop['Area Code']>=5000].Area.unique()
# Define selected regions of interests

regions = ['Eastern Africa', 'Middle Africa', 'Northern Africa', 'Southern Africa', 

           'Western Africa', 'Northern America', 'Central America', 'Caribbean',

           'South America', 'Central Asia', 'Eastern Asia', 'Southern Asia', 

           'South-Eastern Asia', 'Western Asia','Eastern Europe', 'Northern Europe', 

           'Southern Europe','Western Europe','Australia and New Zealand',

           'Melanesia', 'Micronesia', 'Polynesia']



# Define population dataset for these regions

df_pop_region = df_pop[(df_pop['Area Code']>= 5000) &

        (df_pop.Year >= 2015) &

        (df_pop.Area.isin(regions))]



df_pop_region
# Create percent change value for poulation 

df_pop_region['Value_l1'] = df_pop_region.groupby('Area').Value.shift()

df_pop_region['Value_ch'] = (df_pop_region['Value'] 

                             - df_pop_region['Value_l1'])/df_pop_region['Value'] * 100

df_pop_region.sort_values(['Area','Year'])
# Average over years for the population change

df_pop_region_ch = (df_pop_region.groupby('Area')

                     [['Area','Value_ch']].mean()

                     .rename(columns = {'Value_ch':'Pop_ch'}))

df_pop_region_ch
# Repeat the above process for milk consumption total and milk consumption per capita

df_cons_region = df_cons[(df_cons['Area Code']>= 5000) &

                         (df_cons['Item Code'] == 2948) & 

                         (df_cons.Year >= 2015) &

                         (df_cons.Area.isin(regions))]



df_cons_region['Value_l1'] = df_cons_region.groupby('Area').Value.shift()

df_cons_region['Value_ch'] = (df_cons_region['Value'] 

                             - df_cons_region['Value_l1'])/df_cons_region['Value'] * 100



df_cons_region_ch = (df_cons_region.groupby('Area')

                      [['Area','Value_ch']].mean()

                      .rename(columns = {'Value_ch':'Milk_ch'}))

df_cons_region_ch
# Repeat the above process for milk consumption per capita

df_cons_cap_region = df_cons_cap[(df_cons_cap['Area Code'] >= 5000) &

                         (df_cons_cap['Item Code'] == 2948) & 

                         (df_cons_cap.Year >= 2015) &

                         (df_cons_cap.Area.isin(regions))]



df_cons_cap_region
# Define averaged data for population, milk consumption, milk per capita

df_pop_region_avg = (df_pop_region[['Area','Value']].groupby('Area').mean()

                     .rename(columns = {'Value':'Pop'})

                    )

df_cons_region_avg = (df_cons_region[['Area','Value']].groupby('Area').mean()

                      .rename(columns = {'Value':'Milk'})

                     )

df_cons_cap_region_avg = (df_cons_cap_region[['Area','Value']].groupby('Area').mean()

                          .rename(columns = {'Value':'Milk_capita'})

                         )
# Combine data 

df_milk_pop_ch = (df_pop_region_ch

                  .join(df_pop_region_avg)

                  .join(df_cons_region_ch)

                  .join(df_cons_region_avg)

                  .join(df_cons_cap_region_avg)

                 )



# Plot data for Population change vs Milk consumption change

sns.relplot(x = 'Pop_ch', y = 'Milk_ch',  size = 'Milk_capita', 

    data = df_milk_pop_ch 

           ); 
import plotly.express as px

import plotly.graph_objects as go



fct_resize = .75



fig = px.scatter(df_milk_pop_ch[df_milk_pop_ch.Pop > 100e3].reset_index(),

                 x="Pop_ch", y="Milk_ch", text="Area",

                 color = 'Milk_capita', size='Pop', size_max=40 * fct_resize,

                color_continuous_scale=px.colors.sequential.Sunset,

                template='plotly_white')



fig.update_traces(textposition='bottom center', textfont_size=15 * fct_resize)



fig.add_trace(go.Scatter(

    x=[-2, 7],

    y=[-2, 7],

    mode="lines",

    showlegend=False,

    line = dict(color='slategray', width=2 * fct_resize, dash='dot')

))



fig.update_xaxes(range=[-.5,3.1]) 



fig.update_layout(

    height= 600 * fct_resize,

    width = 800 * fct_resize,

    title_text='Fig 3. Changes in Population and Dairy Consumption, 2015-17 Average',

    xaxis_title="Population change, %",

    yaxis_title="Dairy consumption change, %",

    #xaxis_title_font = dict(size=18),

    #yaxis_title_font = dict(size=18),

    font=dict(size=15 * fct_resize),

    coloraxis_colorbar = dict(

    title="Dairy <br> consumption <br> (kg/person)"),# size=14),

   # margin_b=100, #increase the bottom margin to have space for caption

    annotations=[go.layout.Annotation(

        xref='paper',

        yref='paper',

        x= 1, #x= 0.02,

        xanchor= 'right',

        y= 1, #y=-.12,

        yanchor= 'bottom',

        showarrow=False,

        #textfont = dict(size=14),

        text= 'Data source: FAOSTAT.')],

    #textfont = dict(size=14),



    )





fig.show()