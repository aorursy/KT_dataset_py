# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='ISO-8859-1')
df.head()
brazil_df = df[df['Area']=='Brazil']
brazil_df.head()
def products_in_all_years(df):
    product_overs_years = []
    for year in df['Year'].unique():
        aux = df[df['Year']==year]
        product_overs_years.append(aux['Item'].unique())
    return set.intersection(*map(set,product_overs_years))

products = products_in_all_years(brazil_df)
print(len(products))
print(products)
brazil_principal_products = brazil_df[brazil_df['Item'].isin(list(products))]
import plotly.express as px
import plotly.graph_objects as go

def plot_import_export_values_over_year(df):
    import_ = {}
    export_ = {}
    for year in df['Year'].unique():
        export_[year] = 0 
        import_[year] = 0
    for year, value, element in zip(df['Year'], df['Value'], df['Element']):
        if element == 'Export Value':
            export_[year] += value
        elif element == 'Import Value':
            import_[year] += value
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(export_.keys()),
            y=list(export_.values()),
            marker_color='lightsalmon',
            name='Export'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(import_.keys()),
            y=list(import_.values()),
            marker_color='coral',
            name='Import'
        )
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Real news Frequency words'
    })
    
    fig.show()


    
import matplotlib.pyplot as plt
def plot_import_export_quantity_over_year(df):
    import_ = {}
    export_ = {}
    for year in df['Year'].unique():
        export_[year] = 0 
        import_[year] = 0
    for year, value, element in zip(df['Year'], df['Value'], df['Element']):
        if element == 'Export Quantity':
            export_[year] += value
        elif element == 'Import Quantity':
            import_[year] += value
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(export_.keys()),
            y=list(export_.values()),
            marker_color='lightsalmon',
            name='Export'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(import_.keys()),
            y=list(import_.values()),
            marker_color='coral',
            name='Import'
        )
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Real news Frequency words'
    })
    
    fig.show()
plot_import_export_values_over_year(brazil_principal_products)

plot_import_export_quantity_over_year(brazil_principal_products)
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go

brazil_principal_products_import =  brazil_principal_products[brazil_principal_products['Element']=='Import Quantity']
brazil_principal_products_export =  brazil_principal_products[brazil_principal_products['Element']=='Export Quantity']

brazil_principal_products_export_value =  brazil_principal_products[brazil_principal_products['Element']=='Export Value']
brazil_principal_products_import_value =  brazil_principal_products[brazil_principal_products['Element']=='Import Value']

def sum_values(df):
    items =[]
    values = []
    for item in df['Item'].unique():
        items.append(item)
        tmp = df[df['Item']==item]
        v  = tmp['Value'].sum()
        values.append(v)
    return items, values

def plot_bar(df, title, color):
    items, values = sum_values(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=items, y=values, 
                        marker_color=color))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': title
    })
    fig.show()
plot_bar(brazil_principal_products_export, 'Export Quantity (tonnes) products over years', 'turquoise')
plot_bar(brazil_principal_products_import, 'Import Quantity (tonnes) products over years', 'turquoise')
plot_bar(brazil_principal_products_export_value, 'Import Value (tonnes) products over years', 'turquoise')
plot_bar(brazil_principal_products_import_value, 'Export Value (tonnes) products over years', 'turquoise')

import plotly.express as px
brazil_principal_production = brazil_principal_products[brazil_principal_products['Element']=='Production']
fig = px.pie(brazil_principal_production, values='Value', names='Item', title='Brazil Fertilizers production', hole=.7)
fig.update(layout_showlegend=False)
fig.show()
brazil_principal_productions_export_import = brazil_principal_products[(brazil_principal_products['Element']=='Export Value') | (brazil_principal_products['Element']=='Import Value')]
fig = px.pie(brazil_principal_productions_export_import, values='Value', names='Item', title='Total 1000 US busy over the years', hole=.7)
fig.update(layout_showlegend=False)
fig.show()
brazil_principal_productions_export_import = brazil_principal_products[(brazil_principal_products['Element']=='Export Quantity') | (brazil_principal_products['Element']=='Export Quantity')]
fig = px.pie(brazil_principal_productions_export_import, values='Value', names='Item', title='Total Tonners busy over the years', hole=.7)
fig.update(layout_showlegend=False)
fig.show()
brazil_products_non_recussing = brazil_df[~brazil_df['Item'].isin(list(products))]
plot_bar(brazil_products_non_recussing, 'Non-recurring products', 'turquoise')



