# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# import pyplot. 

import matplotlib.pyplot as plt



## in order to show more columns. 

pd.options.display.max_columns = 999



# Any results you write to the current directory are saved as output.a
countries = pd.read_csv('../input/countries-of-the-world/countries of the world.csv')



## From Data Science for good: Kiva funding. 

kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')

loan_themes = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')

mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')

theme_id = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')



## From additional data sources. 

country_stats = pd.read_csv('../input/additional-kiva-snapshot/country_stats.csv')

#all_loans = pd.read_csv('../input/additional-kiva-snapshot/loans.csv')

lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')

loan_coords = pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv')

locations = pd.read_csv('../input/additional-kiva-snapshot/locations.csv')



##mpi

mpi_national = pd.read_csv('../input/mpi/MPI_national.csv')

mpi_subnational = pd.read_csv('../input/mpi/MPI_subnational.csv')



#all_data = [kiva_loans, loan_themes, mpi_region_locations, theme_id, country_stats, loans, lenders, loan_coords, locations]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
## Renaming the columns just how I want it

loan_themes.rename(columns={'Field Partner Name':'field_partner_name',

                            'Loan Theme ID':'loan_theme_id',

                            'Loan Theme Type':'loan_theme_type', 

                            'Partner ID':'partner_id'}, inplace = True)

## renaming the columns just how I like it

theme_id.columns = ['id','loan_theme_id','loan_theme_type','partner_id']

mpi_subnational.columns = ['ISO_country_code',

                           'Country',

                           'Sub_national_region',

                           'world_region',

                           'MPI_national',

                           'MPI_regional',

                           'Headcount_ratio_regional',

                           'intensit_of_deprivation_regional']

kiva_loans.head()
loan_themes.head()
mpi_region_locations.head()
# We're going to be calculating memory usage a lot,

# so we'll create a function to save us some time!

def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)



# We’ll write a loop to iterate over each object column, 

# check if the number of unique values is more than 50%, 

# and if so, convert it to the category atype.

def reduce_by_category_type(df):

    converted_obj = pd.DataFrame()

    for col in df.columns:

        num_unique_values = len(df[col].unique())

        num_total_values = len(df[col])

        if num_unique_values / num_total_values < 0.5 and df[col].dtype == 'object':

            converted_obj.loc[:,col] = df[col].astype('category')

        else:

            converted_obj.loc[:,col] = df[col]

    return converted_obj
kiva_loans = reduce_by_category_type(kiva_loans)

loan_coords = reduce_by_category_type(loan_coords)

loan_themes = reduce_by_category_type(loan_themes)



mpi_national = reduce_by_category_type(mpi_national)



mpi_region_locations  = reduce_by_category_type(mpi_region_locations)



mpi_subnational  = reduce_by_category_type(mpi_subnational)



theme_id = reduce_by_category_type(theme_id)
import plotly.graph_objects as go

from plotly.subplots import make_subplots

counts = go.Bar(

    y=kiva_loans.country.value_counts().head(20).sort_values(ascending = True).index,

    x=kiva_loans.country.value_counts().head(20).sort_values(ascending = True).values,

    orientation = 'h',

    #xaxis = 'Count'

)



temp = kiva_loans.groupby(['country'])['funded_amount'].sum().sort_values(ascending = False).head(20)



donation_amounts = go.Bar(

    y=temp.sort_values(ascending = True).index,

    x=temp.sort_values(ascending = True).values,

    orientation = 'h',

)



fig = make_subplots(rows=1, # row #'s

                          cols=2, # column #'s

                          #specs=[[{'colspan': 2}, None], [{}, {}]],## distribution of chart spacing

                          #shared_yaxes=True, 

                          subplot_titles = ["Countries with Most Loans", 'Countries with Most Funded Amounts']);

#fig.append_trace(data, 1,1);##fig.append_trace(data1,raw #,col #);

fig.append_trace(counts,1,1);

fig.append_trace(donation_amounts,1,2);

fig['layout']['yaxis1'].update(title = 'Country', showgrid = False);                      

#fig['layout']['yaxis3'].update(title = 'Country')



fig['layout']['xaxis1'].update(title = '# of loans')

fig['layout']['xaxis2'].update(title = 'Amount($)')

#fig['layout']['xaxis3'].update(title = 'Count', type = 'log)



fig['layout'].update(height = 800,margin = dict(l = 100,), showlegend = False, title = 'Countries with Most Loan Counts VS Funded Amounts');

#fig.layout.update(title = 'testing')

fig.show()
kiva_loans.replace("CUSCO", "Cusco",inplace=True)



counts = go.Bar(

    y=kiva_loans.region.value_counts().head(20).sort_values(ascending = True).index,

    x=kiva_loans.region.value_counts().head(20).sort_values(ascending = True).values,

    orientation = 'h',

    #xaxis = 'Count'

)



temp = kiva_loans.groupby(['region'])['funded_amount'].sum().sort_values(ascending = False).head(20)



donation_amounts = go.Bar(

    y=temp.sort_values(ascending = True).index,

    x=temp.sort_values(ascending = True).values,

    orientation = 'h',

)



fig = make_subplots(rows=1, # row #'s

                          cols=2, # column #'s

                          #specs=[[{'colspan': 2}, None], [{}, {}]],## distribution of chart spacing

                          #shared_yaxes=True, 

                          subplot_titles = ["Region with Most Loans", 'Region with Most Funded Amounts']);

#fig.append_trace(data, 1,1);##fig.append_trace(data1,raw #,col #);

fig.append_trace(counts,1,1);

fig.append_trace(donation_amounts,1,2);

fig['layout']['yaxis1'].update(title = 'Region', showgrid = False);                      

#fig['layout']['yaxis3'].update(title = 'Country')



fig['layout']['xaxis1'].update(title = '# of loans')

fig['layout']['xaxis2'].update(title = 'Amount($)')

#fig['layout']['xaxis3'].update(title = 'Count', type = 'log)



fig['layout'].update(height = 1000,margin = dict(l=150,), showlegend = False, title = 'Regions with Most Loan Counts VS Funded Amounts');

#fig.layout.update(title = 'testing')

fig.show()
feature = "region"

temp = pd.DataFrame(round(kiva_loans[feature].value_counts(dropna = False, normalize=True)*100,2)).reset_index().rename(columns = {'index':feature,feature:"Percentage of Total"})

temp.dropna(inplace=True)

temp = temp.head(10)

print(" {} ".format(temp.columns[1]).center(40,"*"))

#print ("***** CompanySize *****".center(60, '*') )



for a, b in temp.itertuples(index=False):

    print("{}% loans are given to {}.".format(b, a))

print ('#####')

#print ('27.64 participants did not share an answer for this question')

#print ("Let's find out what they do..")

######

temp = pd.DataFrame(kiva_loans.groupby(['region'])['funded_amount'].sum().sort_values(ascending = False).head(20)).reset_index().rename(columns = {'index':feature,feature:"Percentage of Total Loan"})

temp.dropna(inplace=True)

temp = temp.head(10)

print(" {} ".format(temp.columns[1]).center(40,"*"))

#print ("***** CompanySize *****".center(60, '*') )



for a, b in temp.itertuples(index=False):

    print("{} loans are given to {}.".format(b, a))

#print ('#####')

#print ('27.64 participants did not share an answer for this question')

#print ("Let's find out what they do..")
temp = kiva_loans.sector.value_counts()



data = [go.Bar(

    x=temp.index,

    y=temp.values,

    #width = [0.9,0.9,0.9,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.

    marker = dict(

        #color=['green', 'green', 'green']

    ),

    )]

layout = go.Layout(

    title = "Sectors with Highest Loan Counts",



    xaxis = dict(

        title = "Sectors",

        showgrid = True

    ),

    yaxis = dict( 

        title = '# of loans',

        showgrid = True,

        tickformat = ',d'

    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()



#Alternative way of doing the same thing but really easy.

#sectors = kiva_loans.sector.value_counts().head(30)

#sectors.iplot(kind='bar', xTitle = 'State name', yTitle = "Count", title = 'Top Donor States')
temp = kiva_loans.groupby(['sector'])['funded_amount'].sum().sort_values(ascending= False)

data = go.Bar(

    x = temp.index,

    y = temp.values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'Subject Subcategory Type'

fig.layout.yaxis.title = 'Project Count'

# fig.update_layout(template = 'plotly_dark')

fig.show()



temp = kiva_loans.activity.value_counts().head(20)



data = [go.Bar(

    x=temp.index,

    y=temp.values,

    #width = [1.1, 1.1],## customizing the width.

    marker = dict(

        #color=['green', 'green']

    ),

    )]

layout = go.Layout(

    title = "Top activities of Loans",

    xaxis = dict(

        title = "Activities"

    ),

    yaxis = dict( 

        title = '# of loans', 

        tickformat = ',d'

    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = kiva_loans.repayment_interval.value_counts()



labels = temp.index

values = temp.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']



data = go.Pie(labels=labels, values=values,

               hoverinfo='label+percent', textinfo='percent',

               textfont=dict(size=20),

               marker=dict(colors=colors,

                           line=dict(color='#000000', width=2)))

layout = go.Layout(

    title = "Pie Chart for Repayment Interval",

)



fig = go.Figure(data = [data], layout = layout)

fig.show()
## Most Loans Period in terms of Months.



temp = kiva_loans.term_in_months.value_counts().head(25)



data = [go.Bar(

    x=temp.index,

    y=temp.values,

    #width = [1.1, 1.1],## customizing the width.

    marker = dict(

        color=['green', 'green', 'green']),

    )]

layout = go.Layout(

    title = "Loan period in terms of Months",

    xaxis = dict(

        title = "Activities"

    ),

    yaxis = dict( 

        title = 'Loans', 

        tickformat = ',d'

    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()
## I noticed that some of the data is inconsistant and are basically repeated because of upper/lower case difference. 

kiva_loans.use = kiva_loans.use.str.lower()

## Also I stumbled upon lines where the only difference is a ".". So, I got rid of the difference. 

kiva_loans.use = kiva_loans.use.str.strip('.')

## Its always a good idea to get rid of any extra white spaces. 

kiva_loans.use = kiva_loans.use.str.strip()

kiva_loans.use = kiva_loans.use.str.strip('.')



##There are different version so saying the same thing. therefore I have decided to merge them all together. 

kiva_loans.replace('to buy a water filter to provide safe drinking water for their family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter to provide safe drinking water for her family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter to provide safe drinking water for his family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter to provide safe drinking water for the family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter, to provide safe drinking water for her family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter, to provide safe drinking water for their family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter to provide safe drinking water for their families', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to purchase a water filter to provide safe drinking water for the family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter to provide safe drinking water', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to purchase a water filter to provide safe drinking water', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)

kiva_loans.replace('to buy a water filter in order to provide safe drinking water for their family', 'to buy a water filter to provide safe drinking water for his/her/their family', inplace = True)





##Plotly graphs stats here

temp = kiva_loans.use.value_counts().head(20)



data = [go.Bar( x=temp.index,

    y=temp.values,

    width = [1.0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.

    marker = dict(

        color=['rgb(0, 200, 200)', 'black','black','black','black','black','black','black','black','black','black','black','black']),

    )]

layout = go.Layout(

    title = "Top Uses of the Loans",

    height = 800,

    margin=go.layout.Margin(b =340, r = 250),## this is so we are able to read the labels in xaxis. 'b' stands for bottom, similarly left(l), 

                            ##right(r),top(t) 

    xaxis = dict(

        title = "Uses"

    ),

    yaxis = dict( 

        title = 'Loans',

        tickformat = ',d',



    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()

temp = kiva_loans[kiva_loans.use == "to buy a water filter to provide safe drinking water for his/her/their family"].country.value_counts()



data = go.Bar(

    x = temp.index,

    y = temp.values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'Subject Subcategory Type'

fig.layout.yaxis.title = 'Project Count'

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = kiva_loans[kiva_loans.country == 'Cambodia'].use.value_counts().head(20)



data = [go.Bar(

    x=temp.index,

    y=temp.values,

    width = [1.0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.,

    marker = dict(

        color=['rgb(0, 200, 200)']),

    )]

layout = go.Layout(

    title = "Top Uses of the Loans in Cambodia",

    height = 800,

    margin=go.layout.Margin(b =350, r = 250),## this is so we are able to read the labels in xaxis. 'b' stands for bottom, similarly left(l), 

                            ##right(r),top(t) 

    xaxis = dict(

        title = "Uses"

    ),

    yaxis = dict( 

        title = 'Loans',

        tickformat = ',d',



    )

)

fig = go.Figure(data = data, layout = layout)

fig.show()
temp = kiva_loans[kiva_loans.country == 'Cambodia'].use.value_counts().head(20)

feature = "use"

temp = pd.DataFrame(temp).reset_index().rename(columns = {'index':feature,feature:"Total Loans"})

temp.dropna(inplace=True)

temp = temp.head(10)

print(" {} ".format(temp.columns[1]).center(90,"*"))

#print ("***** CompanySize *****".center(60, '*') )



for a, b in temp.itertuples(index=False):

    print("{} loans are given to {}.".format(b, a))
from wordcloud import WordCloud



loans_cambodia = kiva_loans[kiva_loans.country == 'Cambodia']

names = loans_cambodia["use"][~pd.isnull(kiva_loans["use"])]

#print(names)

wordcloud = WordCloud(max_font_size=90, width=1000, height=300).generate(' '.join(names))

plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.title("Uses of Loans Taken in Cambodia", fontsize=35)

plt.axis("off")

plt.show() 
data = [dict(

        type='choropleth',

        locations= country_stats.country_name,

        locationmode='country names',

        z=country_stats.population,

        text=country_stats.country_name,

#         colorscale='Red',

        marker=dict(line=dict(width=0.7)),

        colorbar=dict(

#             autotick=False, 

            tickprefix='', 

            title='Polulations'),

)]

layout = dict(title = 'Population Map of the world',

             geo = dict(

            showframe = False,

            showcoastlines = False,

            projection = dict(

#                 type = 'Mercatorodes'

            )

        ),)

fig = go.Figure(data=data, layout=layout)

fig.show()
temp = country_stats[['country_name','population']].sort_values(by ='population', ascending = False).head(15)

data = [go.Bar(

    x = temp.country_name,

    y = temp.population,

    marker = dict(line = dict(width = 2))

)]

layout = go.Layout(

    paper_bgcolor='rgb(243, 243, 200)',## asix and lebel background color

    plot_bgcolor='rgb(243, 243, 200)', ## plot background color

    title = 'Top Countries with most people',

    xaxis =dict(title = 'Country'),

    yaxis =dict(title = 'Amount of Population'),

    margin = dict(b = 200)

    

)

fig = go.Figure(data = data, layout =layout)

fig.show()
countries.replace("Congo, Dem. Rep.", 'Democratic Republic of the Congo', inplace=True)

##

countries.rename(columns = {'GDP ($ per capita)':"GDP"}, inplace = True)

countries.rename(columns = {'Pop. Density (per sq. mi.)':"pop_density_per_sq_mile"}, inplace = True)

countries.pop_density_per_sq_mile = countries.pop_density_per_sq_mile.apply(lambda x: int(x.replace(",","")))

countries.Country = countries.Country.apply(lambda x:x.strip())

country_stats = country_stats.merge(countries[['Country',"GDP",'pop_density_per_sq_mile']], left_on ='country_name',right_on='Country', how='left')



##
temp = country_stats[['kiva_country_name','pop_density_per_sq_mile','population','population_below_poverty_line']].sort_values(by = 'population', ascending = False).head(14)

for col in temp.columns:

    temp[col] = temp[col].astype(str)

# writing the text column

temp['text'] = temp['kiva_country_name'] + '<br>' + 'Population Density(per sq/ml): ' + temp['pop_density_per_sq_mile'] + '<br>' + 'Total Populations: ' + temp['population'] + '<br>' + 'Population Below Poverty Line: ' + temp['population_below_poverty_line']

# this is for the size part of the chart

temp['pop_density_per_sq_mile'] = temp['pop_density_per_sq_mile'].astype(float)

#states['text'] = states.apply(lambda x: (states['Donor State']+ '<br>' + "Mean Donations:" + states['Mean Donation Amount'] + '<br>' + "Total Donors:" + states['total_donors'] + "<br" + "Total Donations:" + states['Donation Amount']))

#states.text = states.text.astype(str)

temp['population_below_poverty_line'] = temp['population_below_poverty_line'].astype(float)

trace0 = go.Scatter(

    x=temp.population,

    y=temp.population_below_poverty_line,

    text = temp.text,

    mode='markers',

    marker = {

            'color': temp.population_below_poverty_line.tolist(),

            #'colorscale':"RdYlGn",

            'size': (temp.pop_density_per_sq_mile/80).tolist(),

            'showscale': False, 

            'line': dict(width=2, color='black')

        }

)

layout = go.Layout(

    paper_bgcolor='rgb(243, 243, 200)',## asix and lebel background color

    plot_bgcolor='rgb(243, 243, 200)', ## plot background color

    title = "Population, Population Density and Population Below Poverty Line",

    height = 800,

    xaxis = dict(

        title = "Population"

    ),

    yaxis = dict(

        title = "Population Below Poverty Line"

    )



)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

fig.show()
data = [dict(

        type='choropleth',

        locations= country_stats.country_name,

        locationmode='country names',

        z=country_stats.population_below_poverty_line,

        text=country_stats.country_name,

        marker=dict(

#             colorscale='Red',

            line=dict(width=0.7)),

        colorbar=dict(

#             autotick=False, 

            tickprefix='', 

            title='Polulation<br>below<br>poverty<br>line'),

)]

layout = dict(title = 'Population Below Poverty Line',

             geo = dict(

            showframe = False,

            showcoastlines = False,

            projection = dict(

#                 type = 'Mercatorodes'

            )

        ),)

fig = go.Figure(data=data, layout=layout)

fig.show()
temp = countries[['Country',"GDP"]].sort_values(by = 'GDP', ascending = False)

data = [dict(

        type='choropleth',

        locations= temp.Country,

        locationmode ='country names',

        z=temp.GDP,

        text=temp.Country,

        colorscale='BuGn',

        marker=dict(line=dict(width=0.7)),

        colorbar=dict(

#             autotick=False, 

            tickprefix='', title='GDP<br>Per<br>Capita'),

)]

layout = dict(title = 'World countries with polulations',

             geo = dict(

            showframe = False,

            showcoastlines = False,

            projection = dict(

#                 type = 'Mercatorodes'

            )

        ),)

fig = go.Figure(data=data, layout=layout)

fig.show()
temp = country_stats[['kiva_country_name','pop_density_per_sq_mile',"GDP",'population','population_below_poverty_line']].sort_values(by = 'population', ascending = False).head(14)

for col in temp.columns:

    temp[col] = temp[col].astype(str)

# writing the text column

temp['text'] = temp['kiva_country_name'] + '<br>' +"GDP Per Capita: "+temp["GDP"]+"<br>"+'Population Density(per sq/ml): ' + temp['pop_density_per_sq_mile'] + '<br>' + 'Total Populations: ' + temp['population'] + '<br>' + 'Population Below Poverty Line: ' + temp['population_below_poverty_line']

# this is for the size part of the chart

temp['pop_density_per_sq_mile'] = temp['pop_density_per_sq_mile'].astype(float)

#states['text'] = states.apply(lambda x: (states['Donor State']+ '<br>' + "Mean Donations:" + states['Mean Donation Amount'] + '<br>' + "Total Donors:" + states['total_donors'] + "<br" + "Total Donations:" + states['Donation Amount']))

#states.text = states.text.astype(str)

temp['population'] = temp['population'].astype(float)

trace0 = go.Scatter(

    x=temp.GDP,

    y=temp.population_below_poverty_line,

    text = temp.text,

    mode='markers',

    #colorbar=dict(autotick=False, tickprefix='', title='Polulation<br>below<br>poverty<br>line'),

    marker = {

            'color': temp.population.tolist(),

            #'colorscale':RdYlGn,

            'colorbar' : dict(title = "Population"),

            'size': (temp.pop_density_per_sq_mile/80).tolist(),

            'showscale': True,

            'line': dict(width =2, color ='black')

        }

)

layout = go.Layout(

    paper_bgcolor='rgb(243, 243, 200)',## asix and lebel background color

    plot_bgcolor='rgb(243, 243, 200)', ## plot background color

    title = "Top 15 Populated Countries with Population Density and Population Below Poverty Line",

    height = 800,

    xaxis = dict(

        title = "GDP Per Capita"

    ),

    yaxis = dict(

        title = "Population Below Poverty Line"

    )



)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

fig.show()
## Getting info(population density, GDP, Population and population below poverty line) from country_stats dataframe. 

temp = country_stats[['kiva_country_name','pop_density_per_sq_mile',"GDP",'population','population_below_poverty_line']].sort_values(by = 'population', ascending = False)



## Getting the top 15 funded countries and the funded amount according to Kiva_loans dataset. 

most = kiva_loans.groupby(['country'])['funded_amount'].sum().sort_values(ascending = False).reset_index().head(14)



temp = temp.merge(most, how = 'inner',left_on='kiva_country_name', right_on='country' ).sort_values(by = 'funded_amount', ascending = False)

temp.dropna(inplace=True)                                                                                  





for col in temp.columns:

    temp[col] = temp[col].astype(str)

# writing the text column

temp['text'] = temp['kiva_country_name'] +"<br>"+"Funded Amount: "+temp["funded_amount"]+ '<br>' +"GDP Per Capita: "+temp["GDP"]+"<br>"+'Population Density(per sq/ml): ' + temp['pop_density_per_sq_mile'] + '<br>' + 'Total Populations: ' + temp['population'] + '<br>' + 'Population Below Poverty Line: ' + temp['population_below_poverty_line']

# this is for the size part of the chart

temp['pop_density_per_sq_mile'] = temp['pop_density_per_sq_mile'].astype(float)

#states['text'] = states.apply(lambda x: (states['Donor State']+ '<br>' + "Mean Donations:" + states['Mean Donation Amount'] + '<br>' + "Total Donors:" + states['total_donors'] + "<br" + "Total Donations:" + states['Donation Amount']))

#states.text = states.text.astype(str)

temp['funded_amount']  = temp['funded_amount'].astype(float)

trace0 = go.Scatter(

    x=temp.GDP,

    y=temp.population_below_poverty_line,

    text = temp.text,

    mode='markers',

    #colorbar=dict(autotick=False, tickprefix='', title='Polulation<br>below<br>poverty<br>line'),

    marker = {

            'color': temp.funded_amount.tolist(),

            'colorscale':"Greens",

            'reversescale': True,

            'colorbar' : dict(title = "Funded<br>Amount<br>"),

            'size': (temp.pop_density_per_sq_mile/40).tolist(),

            'showscale': True, 

        'line': dict(width =2, color ='black')

        }

)

layout = go.Layout(

    title = "Top Most Funded Countries by KIVA",

    height = 800,

    xaxis = dict(

        title = "GDP Per Capita"

    ),

    yaxis = dict(

        title = "Polulation Below Poverty Line"

    ),

    paper_bgcolor='rgb(243, 243, 200)',

    plot_bgcolor='rgb(243, 243, 200)',



)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

fig.show()
## Getting info(population density, GDP, Population and population below poverty line) from country_stats dataframe. 

temp = country_stats[['kiva_country_name','pop_density_per_sq_mile',"GDP",'population','population_below_poverty_line']].sort_values(by = 'population', ascending = False)



## Getting the top 15 funded countries and the funded amount according to Kiva_loans dataset. 

least = kiva_loans.groupby(['country'])['funded_amount'].sum().sort_values(ascending = False).reset_index().tail(20)



temp = temp.merge(least, how = 'inner',left_on='kiva_country_name', right_on='country' ).sort_values(by = 'funded_amount', ascending = False)

                                                                                            





for col in temp.columns:

    temp[col] = temp[col].astype(str)

# writing the text column

temp['text'] = temp['kiva_country_name'] +"<br>"+"Funded Amount: "+temp["funded_amount"]+ '<br>' +"GDP Per Capita: "+temp["GDP"]+"<br>"+'Population Density(per sq/ml): ' + temp['pop_density_per_sq_mile'] + '<br>' + 'Total Populations: ' + temp['population'] + '<br>' + 'Population Below Poverty Line: ' + temp['population_below_poverty_line']

# this is for the size part of the chart

temp['pop_density_per_sq_mile'] = temp['pop_density_per_sq_mile'].astype(float)

#states['text'] = states.apply(lambda x: (states['Donor State']+ '<br>' + "Mean Donations:" + states['Mean Donation Amount'] + '<br>' + "Total Donors:" + states['total_donors'] + "<br" + "Total Donations:" + states['Donation Amount']))

#states.text = states.text.astype(str)

temp['funded_amount'] = temp['funded_amount'].astype(float)

temp.dropna(inplace = True)

trace0 = go.Scatter(

    x=temp.GDP,

    y=temp.population_below_poverty_line,

    text = temp.text,

    mode='markers',

    #colorbar=dict(autotick=False, tickprefix='', title='Polulation<br>below<br>poverty<br>line'),

    marker = {

            'color': temp.funded_amount.tolist(),

            'colorscale':"Greens",

            'reversescale': True,

            'colorbar' : dict(title = "Funded<br>Amount<br>"),

            'size': (temp.pop_density_per_sq_mile/40).tolist(),

            'showscale': True, 

        'line': dict(width =2, color ='black')

        }

)

layout = go.Layout(

    title = "Least Funded Countries by KIVA",

    height = 800,

    xaxis = dict(

        title = "GDP Per Capita"

    ),

    yaxis = dict(

        title = "Polulation Below Poverty Line"

    ),

    paper_bgcolor='rgb(243, 243, 200)',

    plot_bgcolor='rgb(243, 243, 200)',



)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

fig.show()
## Getting info(population density, GDP, Population and population below poverty line) from country_stats dataframe. 

temp = country_stats[['kiva_country_name','pop_density_per_sq_mile',"GDP",'population','population_below_poverty_line']].sort_values(by = 'population', ascending = False)



## Getting the top 15 funded countries and the funded amount according to Kiva_loans dataset. 

most = kiva_loans.groupby(['country'])['funded_amount'].sum().sort_values(ascending = False).reset_index()



temp = temp.merge(most, how = 'inner',left_on='kiva_country_name', right_on='country' ).sort_values(by = 'funded_amount', ascending = False)

                                                                                     



for col in temp.columns:

    temp[col] = temp[col].astype(str)

# writing the text column

temp['text'] = temp['kiva_country_name'] +"<br>"+"Funded Amount: "+temp["funded_amount"]+ '<br>' +"GDP Per Capita: "+temp["GDP"]+"<br>"+'Population Density(per sq/ml): ' + temp['pop_density_per_sq_mile'] + '<br>' + 'Total Populations: ' + temp['population'] + '<br>' + 'Population Below Poverty Line: ' + temp['population_below_poverty_line']

# this is for the size part of the chart

temp['pop_density_per_sq_mile'] = temp['pop_density_per_sq_mile'].astype(float)

#states['text'] = states.apply(lambda x: (states['Donor State']+ '<br>' + "Mean Donations:" + states['Mean Donation Amount'] + '<br>' + "Total Donors:" + states['total_donors'] + "<br" + "Total Donations:" + states['Donation Amount']))

#states.text = states.text.astype(str)

temp['funded_amount'] = temp['funded_amount'].astype(float)

temp.dropna(inplace = True)    





trace0 = go.Scatter(

    x=temp.GDP,

    y=temp.population_below_poverty_line,

    text = temp.text,

    mode='markers',

    #colorbar=dict(autotick=False, tickprefix='', title='Polulation<br>below<br>poverty<br>line'),

    marker = {

            'color': temp.funded_amount.tolist(),

            'colorscale':"Greens",

            'reversescale': True,

            'colorbar' : dict(title = "Funded<br>Amount<br>"),

            'size': (temp.pop_density_per_sq_mile/40).tolist(),

            'showscale': True,

            'line': dict(width = 2, color = 'black')

        }

)

layout = go.Layout(

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    title = "Countries Funded by KIVA",

    height = 800,

    xaxis = dict(

        title = "GDP Per Capita",

        gridcolor='rgb(255, 255, 255)',

        type = 'log',

        ticklen = 5,

        gridwidth =2,

    ),

    yaxis = dict(

        title = "Polulation Below Poverty Line",

        gridcolor='rgb(255, 255, 255)',

        ticklen = 5, 

        gridwidth =2,

    )



)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

fig.show()
import math

## Getting info(population density, GDP, Population and population below poverty line) from country_stats dataframe. 

temp = country_stats[['kiva_country_name','continent','pop_density_per_sq_mile',"GDP",'population','population_below_poverty_line']].sort_values(by = 'population', ascending = False)



## Getting the top 15 funded countries and the funded amount according to Kiva_loans dataset. 

most = kiva_loans.groupby(['country'])['funded_amount'].sum().sort_values(ascending = False).reset_index()



## merging funded amount with temp dataframe. 

temp = temp.merge(most, how = 'inner',left_on='kiva_country_name', right_on='country' ).sort_values(by = 'funded_amount', ascending = False)

temp = temp.sort_values(['continent', 'kiva_country_name'])



slope = 2.666051223553066e-05

hover_text = []

bubble_size = []



for index, row in temp.iterrows():

    hover_text.append(('Country: {kiva_country_name}<br>'+

                      'Funded Amount: {funded_amount}<br>'+

                      'GDP per capita: {GDP}<br>'+

                      'Population: {population}<br>'+

                      'Population Density: {pop_density_per_sq_mile}').format(kiva_country_name=row['kiva_country_name'],

                                            funded_amount=row['funded_amount'],

                                            GDP=row['GDP'],

                                            population=row['population'],

                                            pop_density_per_sq_mile=row['pop_density_per_sq_mile']))

    bubble_size.append(math.sqrt(row['population']*slope))



temp['text'] = hover_text

temp['size'] = bubble_size

sizeref = 2.*max(temp['size'])/(100**2)



trace0 = go.Scatter(

    x=temp['GDP'][temp['continent'] == 'Africa'],

    y=temp['funded_amount'][temp['continent'] == 'Africa'],

    mode='markers',

    name='Africa',

    text=temp['text'][temp['continent'] == 'Africa'],

    marker=dict(

        symbol='circle',

        sizemode='area',

        sizeref=sizeref,

        size=temp['size'][temp['continent'] == 'Africa'],

        line=dict(

            width=2

        ),

    )

)

trace1 = go.Scatter(

    x=temp['GDP'][temp['continent'] == 'Americas'],

    y=temp['funded_amount'][temp['continent'] == 'Americas'],

    mode='markers',

    name='Americas',

    text=temp['text'][temp['continent'] == 'Americas'],

    marker=dict(

        symbol='circle',

        sizemode='area',

        sizeref=sizeref,

        size=temp['size'][temp['continent'] == 'Americas'],

        line=dict(

            width=2

        ),

    )

)

trace2 = go.Scatter(

    x=temp['GDP'][temp['continent'] == 'Asia'],

    y=temp['funded_amount'][temp['continent'] == 'Asia'],

    mode='markers',

    name='Asia',

    text=temp['text'][temp['continent'] == 'Asia'],

    marker=dict(

        symbol='circle',

        sizemode='area',

        sizeref=sizeref,

        size=temp['size'][temp['continent'] == 'Asia'],

        line=dict(

            width=2

        ),

    )

)

trace3 = go.Scatter(

    x=temp['GDP'][temp['continent'] == 'Europe'],

    y=temp['funded_amount'][temp['continent'] == 'Europe'],

    mode='markers',

    name='Europe',

    text=temp['text'][temp['continent'] == 'Europe'],

    marker=dict(

        symbol='circle',

        sizemode='area',

        sizeref=sizeref,

        size=temp['size'][temp['continent'] == 'Europe'],

        line=dict(

            width=2

        ),

    )

)

trace4 = go.Scatter(

    x=temp['GDP'][temp['continent'] == 'Oceania'],

    y=temp['funded_amount'][temp['continent'] == 'Oceania'],

    mode='markers',

    name='Oceania',

    text=temp['text'][temp['continent'] == 'Oceania'],

    marker=dict(

        symbol='circle',

        sizemode='area',

        sizeref=sizeref,

        size=temp['size'][temp['continent'] == 'Oceania'],

        line=dict(

            width=2

        ),

    )

)



data = [trace0, trace1, trace2, trace3, trace4]

layout = go.Layout(

    title='Funded Amount v. GDP Per Capita',

    xaxis=dict(

        title='GDP per capita',

        gridcolor='rgb(255, 255, 255)',

        #range=[2.003297660701705, 5.191505530708712],

        type='log', ## spreads the points throughout the plot using log in the axis

        zerolinewidth=1,

        ticklen=5,

        gridwidth=2,

    ),

    yaxis=dict(

        title='Funded Amount',

        gridcolor='rgb(255, 255, 255)',

        #range=[36.12621671352166, 91.72921793264332],

        zerolinewidth=1,

        ticklen=5,

        gridwidth=2,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

)



fig = go.Figure(data=data, layout=layout)

fig.show()
country_stats['education_index'] = ((country_stats.expected_years_of_schooling+country_stats.mean_years_of_schooling)/2)



country_stats.dropna(inplace=True)

data = [dict(

        x = country_stats.life_expectancy.tolist(),

        y = country_stats.population_below_poverty_line.tolist(),

        mode = 'markers',

        text = country_stats.country_name,

        marker = {

            'color': country_stats.population.tolist(),

            'colorscale':"Viridis",

            'size': (country_stats.hdi*30).tolist(),

            'showscale': True

        })]

    

layout = go.Layout(

    title = "Human Development Index(HDI), Education Index(EI), Income Index(GNI) and Life Expectancy Index(LEI)",

    height = 1200,

    xaxis = dict(

        title = "Life Expectency"

    ),

    yaxis = dict(

        title = "GNI Index"

    )



)



fig = go.Figure(data = data, layout = layout)

fig.show()
temp = pd.DataFrame(kiva_loans.borrower_genders.dropna().str.split(",").tolist()).stack().value_counts().reset_index()

temp.rename(columns={'index':'gender', 0:'total'}, inplace=True)

temp.gender = temp.gender.apply(lambda x: x.strip())

temp = temp.groupby(['gender'])['total'].sum().reset_index()



labels = temp.gender

values = temp.total

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']



data = go.Pie(labels=labels, values=values,

               hoverinfo='label+percent', textinfo='percent',

               textfont=dict(size=20),

               marker=dict(colors=colors,

                           line=dict(color='#000000', width=2)))

layout = go.Layout(

    title = "Pie Chart for Repayment Interval",

)



fig = go.Figure(data = [data], layout = layout)

fig.show()