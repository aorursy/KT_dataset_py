import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
df = pd.read_csv("../input/nyc-rolling-sales.csv")
df.head()
df.columns = ['Unnamed: 0', 'borough', 'neighborhood','building_class category','tax_class_at_present', 'block', 'lot', 'ease_ment','building_class_at_present', 'address', 'apartment_number', 'zip_code',
       'residential_units', 'commercial_units', 'total_units','land_square_feet', 'gross_square_feet', 'year_built','tax_class_at_time_of_sale', 'building_class_at_time_of_sale',
       'sale_price', 'sale_date']
df.info()
# deleting the Unnamed column
del df['Unnamed: 0']

# SALE PRICE, LAND and GROSS SQUARE FEET is object type but should be numeric data type columns hence converting them to numeric
df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
df['land_square_feet'] = pd.to_numeric(df['land_square_feet'], errors='coerce')
df['gross_square_feet']= pd.to_numeric(df['gross_square_feet'], errors='coerce')

# Both TAX CLASS attributes should be categorical
df['tax_class_at_time_of_sale'] = df['tax_class_at_time_of_sale'].astype('category')
df['tax_class_at_present'] = df['tax_class_at_present'].astype('category')
#SALE DATE is object but should be datetime
df['sale_date']    = pd.to_datetime(df['sale_date'], errors='coerce')
df['sale_year']    = df['sale_date'].dt.year
df['sale_month']   = df['sale_date'].dt.month
df['sale_quarter'] = df['sale_date'].dt.quarter
df['sale_day']     = df['sale_date'].dt.day
df['sale_weekday'] = df['sale_date'].dt.weekday
# as seen in the other Kernels there are some duplicate values that need to be deleted 
# lets check and delete those values
print("Number of duplicates in the given dataset = {0}".format(sum(df.duplicated(df.columns))))
df = df.drop_duplicates(df.columns, keep='last')
print("Number of duplicates in the given dataset after cleanup = {0}".format(sum(df.duplicated(df.columns))))
# lets find out the percentage of non null values in each column and plot them to get a better view
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(df), 2)
dataa = [go.Bar(
            y= df.columns,
            x = count_pct,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Percentage of non-null values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')
# as there are rows where sale price is null, we should remove them from our dataset
df = df[df['sale_price'].notnull()]
df = df[(df['sale_price'] > 100000) & (df['sale_price'] < 5000000)]

# Removes all NULL values
df = df[df['land_square_feet'].notnull()] 
df = df[df['gross_square_feet'].notnull()] 

# Keeps properties with fewer than 20,000 Square Feet, which is about 2,000 Square Metres
df = df[df['gross_square_feet'] < 20000]
df = df[df['land_square_feet'] < 20000]
# lets find out the percentage of non null values in each column and plot them to get a better view
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(df), 2)

dataa = [go.Bar(
            y= df.columns,
            x = count_pct,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Percentage of non-null values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    yaxis=dict(
        title='column'
    )
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')
print("Length of dataset after cleanup = {0}".format(len(df)))
# Create a trace
trace = go.Scatter(
    x = np.sort(df['sale_price']),
    y = np.arange(len(df)),
    mode = 'markers'
)
layout = go.Layout(
    title='Sale Prices',
    autosize = True,
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)
d = [trace]

# Plot and embed in ipython notebook!
fig = go.Figure(data=d, layout = layout)
py.iplot(fig, filename='basic-scatter')
trace1 = go.Histogram(
    x = df['sale_price'],
    name = 'sale_price'
)
dat = [trace1]
# Plot!
#py.iplot(dat, filename='Distplot with Normal Curve')
from scipy.stats import skew
print("Skewness of Sale Price attribute is : {0}".format(skew(df['sale_price'])))

trace2 = go.Histogram(
    x = np.log(df['sale_price']), 
    name = 'log(sale_price)'
)
dat = [trace1]
# Plot!
#py.iplot(dat, filename='Distplot with Normal Curve')
print("Skewness of Sale Price attribute after applying log is : {0}".format(skew(np.log(df['sale_price']))))

fig = tls.make_subplots(rows=2, cols=1, subplot_titles=('Sale Prices', 'Sale Prices after applying log'));
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 2, 1);
fig['layout'].update(height=600, width=800,title='Histogram Plot for Sale prices of Houses');
py.iplot(fig, filename='simple-subplot')

# creating a different new copy of dataset
df2 = pd.read_csv("../input/nyc-rolling-sales.csv")
del df2['Unnamed: 0']
df2.columns = ['borough', 'neighborhood','building_class_category','tax_class_at_present', 'block', 'lot', 'ease_ment','building_class_at_present', 'address', 'apartment_number', 'zip_code',
       'residential_units', 'commercial_units', 'total_units','land_square_feet', 'gross_square_feet', 'year_built','tax_class_at_time_of_sale', 'building_class_at_time_of_sale',
       'sale_price', 'sale_date']
# lets rename boroughs and do some visualization on it
df2['borough'][df2['borough'] == 1] = 'Manhattan'
df2['borough'][df2['borough'] == 2] = 'Bronx'
df2['borough'][df2['borough'] == 3] = 'Brooklyn'
df2['borough'][df2['borough'] == 4] = 'Queens'
df2['borough'][df2['borough'] == 5] = 'Staten Island'

df2['sale_price'] = pd.to_numeric(df2['sale_price'], errors='coerce')
df2['land_square_feet'] = pd.to_numeric(df2['land_square_feet'], errors='coerce')
df2['gross_square_feet']= pd.to_numeric(df2['gross_square_feet'], errors='coerce')

# Both TAX CLASS attributes should be categorical
df2['tax_class_at_time_of_sale'] = df2['tax_class_at_time_of_sale'].astype('category')
df2['tax_class_at_present'] = df2['tax_class_at_present'].astype('category')
# distribution of houses in each borough
boroughs = ['Manhattan','Bronx','Brooklyn','Queens','Staten Island']
property_count = []
for b in boroughs:
    property_count.append(len(df2.borough[df2.borough == b]))

fig = {
  "data": [
    {
      "values": property_count,
      "labels": boroughs,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Percentage of Properties in Boroughs",
    }
}
py.iplot(fig, filename='donut')


dataa = [go.Bar(
            y= boroughs,
            x = property_count,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Number of Housing Properties in each Bourough',
    autosize = False,
    width=800,
    height=500,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')
# average price of a house in each borough
# box plot
df2 = df2[df2['sale_price'].notnull()]
df2 = df2[(df2['sale_price'] > 100000) & (df2['sale_price'] < 5000000)]

trace0 = go.Box(
    y=df2.sale_price[df2.borough == 'Manhattan' ],
    name = 'Manhattan',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df2.sale_price[df2.borough ==  'Bronx' ],
    name = 'Bronx',
    marker = dict(
        color = 'rgb(8,81,156)',
    )
)
trace2 = go.Box(
    y=df2.sale_price[df2.borough ==  'Brooklyn' ],
    name = 'Brooklyn',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace3 = go.Box(
    y=df2.sale_price[df2.borough ==  'Queens' ],
    name = 'Queens',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
trace4 = go.Box(
    y=df2.sale_price[df2.borough ==  'Staten Island' ],
    name = 'Staten Island',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

dat = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(
    title='Housing Prices in Boroughs',
    xaxis=dict(
        title='Borough'
    ),
    yaxis=dict(
        title='Sale Price'
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout=layout)
py.iplot(fig)

# Rainbow plot
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 12)]
data = [{
    'y' : df.sale_price[df.sale_month == ind],
    'type':'box',
    'name' : months[ind - 1],
    'marker':{'color': colors[ind - 1]}
} for ind in range(1,13)]

layout = go.Layout(
    title='Housing Prices in each Borough',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='Sale Price'
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=data, layout=layout)
#dat = [trace0, trace1, trace2, trace3, trace4]
py.iplot(fig)

# average LAND SQUARE FEET in each borough box plot
d3 = pd.DataFrame(df.groupby(['borough']).mean()).reset_index()
d3['borough'][d3.borough == 1] = 'Manhattan'
d3['borough'][d3.borough == 2] = 'Bronx'
d3['borough'][d3.borough == 3] = 'Brooklyn'
d3['borough'][d3.borough == 4] = 'Queens'
d3['borough'][d3.borough == 5] = 'Staten Island'
total = d3.land_square_feet.sum()
trace0 = go.Scatter(
    x=d3.borough,
    y=d3.land_square_feet,
    mode='markers',
    marker=dict(
        size=[((x/total)*300) for x in d3.land_square_feet],
        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)', 'rgb(255, 15, 54)'],
    )
)

data = [trace0]
layout = go.Layout(
    title='Average Land square feet of properties in each Borough',
    xaxis=dict(
        title='Borough',
        gridcolor='rgb(255, 255, 255)',
    ),
    yaxis=dict(
        title='Land square feet',
        gridcolor='rgb(255, 255, 255)',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bubblechart-size')
# average GROSS SQUARE FEET in each borough box plot
total = d3.gross_square_feet.sum()
trace0 = go.Scatter(
    x=d3.borough,
    y=d3.gross_square_feet,
    mode='markers',
    marker=dict(
        size=[((x/total)*300) for x in d3.gross_square_feet],
        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)', 'rgb(255, 15, 54)'],
    )
)

layout = go.Layout(
    title='Average Gross square feet in each Borough',
    xaxis=dict(
        title='Boroughs',
    ),
    yaxis=dict(
        title='Average gross square feet',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)


data = [trace0]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='gross-square-feet-borough-bubble-chart')
p = pd.DataFrame(df.groupby(['borough', 'sale_month']).sale_price.count()).reset_index()
colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 12)]
data = [
    {
        'x': months,
        'y': [boroughs[ind]] * 12,
        'showlegend' : False,
        'mode': 'markers',
        'text': [ x for x in p.sale_price[p.borough == ind+1]],
        'marker': {
            'color': colors,
            'size': [(x/np.sum(p.sale_price[p.borough == ind+1])*400) for x in p.sale_price[p.borough == ind+1]],
        }
    } for ind in range(5)
]

layout = go.Layout(
    title='Number of properties sold in each borough in each month',
    xaxis=dict(
        title='Month',
        gridcolor='rgb(255, 255, 255)',
    ),
    yaxis=dict(
        title='Borough',
        gridcolor='rgb(255, 255, 255)',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='temp')

from collections import Counter
# print(Counter(data2['BUILDING CLASS CATEGORY']))
c = dict(Counter(df2['building_class_category']))
import operator
sorted_category = sorted(c.items(), key=operator.itemgetter(1))
cat_name  = []
cat_value = []
for tup in sorted_category:
    cat_name.append(tup[0])
    cat_value.append(tup[1])
# plot to see what class do majority of the buildings belong to 
dataa = [go.Bar(
            y= cat_name,
            x = cat_value,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='What building class do majority of apartments belong to?',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=350,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    xaxis=dict(
        title='Number of housing properties',
    ),
    yaxis=dict(
        title='Building class',
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')
tax_class = ["tax_class_1","tax_class_2","tax_class_4"]
classes = ["1","2","4"]
data = []
for i in range(3):
    trace = {
            "type": 'violin',
            "x": tax_class[i],
            "y": df.sale_price[df.tax_class_at_present.str.contains(classes[i])],
            "name": tax_class[i],
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)

        
fig = {
    "data": data,
    "layout" : {
        "title": "Sale Prices of houses according to tax class",
        "yaxis": {
            "title": "Sale Price",
            "zeroline": False,
        },
        "xaxis": {
            "title": "Tax classes"
        }
    }
}


py.iplot(fig, filename='tax-class-sale-price-voilin-plots', validate = False)
from collections import Counter
neighborhoods = list(dict(Counter(df.neighborhood).most_common(20)).keys())

avg_sale_prices = []
for i in neighborhoods:
    avg_price = np.mean(df.sale_price[df.neighborhood == i])
    avg_sale_prices.append(avg_price)
    
dataa = [go.Bar(
            y= neighborhoods,
            x = avg_sale_prices,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]
layout = go.Layout(
    title='Average House Price in the top 20 neighborhoods',
    autosize = True,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    xaxis=dict(
        title='Sale Price',
    ),
    yaxis=dict(
        title='Neighborhood',
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='barplottype')
# Compute the correlation matrix
d= df[['sale_price', 'total_units','gross_square_feet',  'land_square_feet', 'residential_units', 
         'commercial_units', 'borough', 'block', 'lot', 'zip_code', 'year_built',]]
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, 
            square=True, linewidths=.5, annot=True, cmap=cmap)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of all Numerical Variables')
plt.show()