# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

# import pyplot. 
import matplotlib.pyplot as plt

## in order to show more columns. 
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.a
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
kiva_loans.head()
total = kiva_loans.isnull().sum()[kiva_loans.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(kiva_loans)*100,4))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
loan_themes.head()
total = loan_themes.isnull().sum()[loan_themes.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(loan_themes)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
mpi_region_locations.head()
total = mpi_region_locations.isnull().sum()[mpi_region_locations.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(mpi_region_locations)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
theme_id.columns = ['id','loan_theme_id','loan_theme_type','partner_id']
theme_id.head()
total = theme_id.isnull().sum()[theme_id.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(theme_id)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
country_stats.head()
total = country_stats.isnull().sum()[country_stats.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(country_stats)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
lenders.head()
total = lenders.isnull().sum()[lenders.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(lenders)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
loan_coords.head()
total = loan_coords.isnull().sum()[loan_coords.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(loan_coords)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
locations.head()
total = locations.isnull().sum()[locations.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(locations)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
mpi_national.head()
total = mpi_national.isnull().sum()[mpi_national.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(mpi_national)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
mpi_subnational.columns = ['ISO_country_code','Country','Sub_national_region','world_region','MPI_national','MPI_regional','Headcount_ratio_regional','intensit_of_deprivation_regional']
mpi_subnational.head()
total = mpi_subnational.isnull().sum()[mpi_subnational.isnull().sum() != 0].sort_values(ascending = False)
percent = pd.Series(round(total/len(mpi_subnational)*100,2))
pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
top_countries = kiva_loans.country.value_counts().head(20)

data = [go.Bar(
    x=top_countries.index,
    y=top_countries.values,
    width = [1.1],## customizing the width.
    marker = dict(
        color=['green',]),
    )]
layout = go.Layout(
    title = "Countries with Most Loans",
    xaxis = dict(
        title = "Countries"
    ),
    yaxis = dict(
        title = 'Loans',
        autorange = True,
        autotick = True,
        showgrid = True,
        showticklabels = True,
        tickformat = ',d'### Note for me: took me about an hour just to get this right. 
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='basic-bar')
top_funded_countries = kiva_loans.groupby(["country"])["funded_amount"].sum().sort_values(ascending = False).head(20)
##top_funded_countries_2 = pd.pivot_table(kiva_loans, index = 'country',values = 'funded_amount', aggfunc='sum').sort_values(by = "funded_amount", ascending = False).head(20) ##Alternative way to get the same info. 

data = [go.Bar(
    x=top_funded_countries.index, ## top_funded_countries_2.index
    y=top_funded_countries.values, ## top_funded_countries_2.funded_amount.values
    width = [1.1,],## customizing the width.
    marker = dict(
        color=['green',]),##makes the first bar green. 
    )]
layout = go.Layout(
    title = "Countries with Highest Funded Amounts",
    margin = go.Margin(b = 140,l = 95),
    xaxis = dict(
        title = "Countries"
    ),
    yaxis = dict(
        title = '$ amount',
        showgrid = True,
        ticks = 'inside'
        
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='basic-bar')
top_regions = kiva_loans.region.value_counts().head(20)

data = [go.Bar(
    x=top_regions.index,
    y=top_regions.values,
    #width = [1.0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.
    marker = dict(
        color=['green']),
    )]
layout = go.Layout(
    title = "Top Regions for Kiva Loans",
    margin = go.Margin(b = 150),
    xaxis = dict(
        title = "Regions"
    ),
    yaxis = dict( 
        title = 'Loan Counts',
        tickformat = ',d',
        ticks = 'inside'
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='basic-bar')
sectors = kiva_loans.sector.value_counts()

data = [go.Bar(
    x=sectors.index,
    y=sectors.values,
    #width = [0.9,0.9,0.9,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.
    marker = dict(
        color=['green', 'green', 'green']),
    )]
layout = go.Layout(
    title = "Sectors with Highest Loan Counts",
    xaxis = dict(
        title = "Sectors"
    ),
    yaxis = dict( 
        title = 'Loans',
        tickformat = ',d'
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='basic-bar')
activities = kiva_loans.activity.value_counts().head(20)

data = [go.Bar(
    x=activities.index,
    y=activities.values,
    #width = [1.1, 1.1],## customizing the width.
    marker = dict(
        color=['green', 'green']),
    )]
layout = go.Layout(
    title = "Top activities of Loans",
    xaxis = dict(
        title = "Activities"
    ),
    yaxis = dict( 
        title = 'Loans', 
        tickformat = ',d'
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename='basic-bar')
repayment = kiva_loans.repayment_interval.value_counts()

labels = repayment.index
values = repayment.values
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
py.iplot(fig, filename='styled_pie_chart')
## Most Loans Period in terms of Months.

terms = kiva_loans.term_in_months.value_counts()

data = [go.Bar(
    x=terms.index,
    y=terms.values,
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
py.iplot(fig, filename='basic-bar')
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
uses = kiva_loans.use.value_counts().head(20)

data = [go.Bar(
    x=uses.index,
    y=uses.values,
    width = [1.0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.
    marker = dict(
        color=['rgb(0, 200, 200)', 'black)','black','black','black','black','black','black','black','black','black','black','black']),
    )]
layout = go.Layout(
    title = "Top Uses of the Loans",
    margin=go.Margin(b =270),## this is so we are able to read the labels in xaxis. 'b' stands for bottom, similarly left(l), 
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
py.iplot(fig, filename='horizontal-bar')
kiva_loans[kiva_loans.use == "to buy a water filter to provide safe drinking water for his/her/their family"].country.value_counts()
loans_cambodia = kiva_loans[kiva_loans.country == 'Cambodia']
uses = loans_cambodia.use.value_counts().head(20)

data = [go.Bar(
    x=uses.index,
    y=uses.values,
    width = [1.0,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],## customizing the width.
    marker = dict(
        color=['rgb(0, 200, 200)']),
    )]
layout = go.Layout(
    title = "Top Uses of the Loans in Cambodia",
    margin=go.Margin(b =270),## this is so we are able to read the labels in xaxis. 'b' stands for bottom, similarly left(l), 
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
py.iplot(fig, filename='horizontal-bar')
from wordcloud import WordCloud

names = loans_cambodia["use"][~pd.isnull(kiva_loans["use"])]
#print(names)
wordcloud = WordCloud(max_font_size=90, width=800, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("What Drives People to Take Loans", fontsize=35)
plt.axis("off")
plt.show() 
from wordcloud import WordCloud

names = kiva_loans["use"][~pd.isnull(kiva_loans["use"])]
#print(names)
wordcloud = WordCloud(max_font_size=60, width=800, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("What Drives People to Take Loans", fontsize=35)
plt.axis("off")
plt.show() 
## changing date type from object to datetime64 for date column.  
print ("Given raised_time type: {}".format(kiva_loans.posted_time.dtype))
## modifying the date type so that we can access day, month or yeah for analysis.
kiva_loans['posted_time'] = pd.to_datetime(kiva_loans['posted_time'], format = "%Y-%m-%d %H:%M:%S", errors='ignore')
kiva_loans['disbursed_time'] = pd.to_datetime(kiva_loans['disbursed_time'], format = "%Y-%m-%d %H:%M:%S", errors='ignore')
kiva_loans['funded_time'] = pd.to_datetime(kiva_loans['funded_time'], format = "%Y-%m-%d %H:%M:%S", errors='ignore')
print ("Modified raised_time type: {}".format( kiva_loans.posted_time.dtype))

loan_each_day_of_month = kiva_loans.funded_time.dt.day.value_counts().sort_index()

data = [go.Scatter(
    x=loan_each_day_of_month.index,
    y=loan_each_day_of_month.values,
    mode = 'lines+markers'
)]
layout = go.Layout(
    title = "Loan Funded in Each Day of Every Month",
    xaxis = dict(
        title = "Dates",
        autotick = False
    ),
    yaxis = dict(
        title = 'Loans',
        tickformat = ',d'
    )
    
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'scatter_funded_amount')
loan_each_month = kiva_loans.funded_time.dt.month.value_counts().sort_index()

data = [go.Scatter(
    x=loan_each_month.index,
    y=loan_each_month.values,
    mode = 'lines+markers'
)]
layout = go.Layout(
    title = "Loan Funded in Each Month",
    xaxis = dict(
        title = "Months",
        autotick = False
    ),
    yaxis = dict(
        title = 'Loans',
        tickformat = ',d'
    )
    
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'scatter_funded_amount')


loan_each_year = kiva_loans.funded_time.dt.year.value_counts().sort_index()

data = [go.Scatter(
    x=loan_each_year.index,
    y=loan_each_year.values,
    mode = 'lines+markers'
)]
layout = go.Layout(
    title = "Loan Funded in Each Year",
    xaxis = dict(
        title = "years",
        autotick = False
    ),
    yaxis = dict(
        title = 'loans',
        tickformat = ',d'
    )
    
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'scatter_funded_amount')
## Let's Dive into Philippines
phili_kiva = kiva_loans[kiva_loans.country == "Philippines"]
print ("Total loans: {}".format(len(phili_kiva)))
print("Total funded amount: ${}".format(phili_kiva.funded_amount.sum()))
print("Total loan amount: ${}".format(phili_kiva.loan_amount.sum()))
loan_per_day = phili_kiva.date.value_counts(sort = False)

data = [go.Scatter(
    x=loan_per_day.index,
    y=loan_per_day.values,
    mode = 'markers'
)]
layout = go.Layout(
    title = "Kiva funding in Philippines",
    xaxis = dict(
        title = "date"
    ),
    yaxis = dict(
        title = '$ amount'
    )
    
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'scatter_funded_amount')
phili_kiva.groupby(['sector'])['funded_amount'].mean().sort_values(ascending = False)
## Merging two columns on "loans.loan_id = loan_coords.loan_id " 
kiva_loans = kiva_loans.merge(loan_coords, how = 'left', left_on = 'id' ,right_on ='loan_id' )
## I am really into working with geo data. So, Let's see if we are missing any lon/lats at this point.
missing_geo = kiva_loans[kiva_loans.latitude.isnull()]
print ("Total missing Latitudes and Longtitudes are: {}".format(len(missing_geo)))
## Extracting region from formatted_address column. 
locations['region'] = [i.split(',')[0] for i in locations.formatted_address]

## Extracting country names from formatted_address column. 
## List comprehension doesn't know how to deal with errors, So, I have had to use good and old for loop. 
a = []
for i in locations.formatted_address:
    try:
        a.append((i.split(',')[1]).strip())
    except:
        a.append((i.split(',')[0]).strip())
        
locations['country'] = a
