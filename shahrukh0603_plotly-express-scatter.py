# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px
#data from https://github.com/devbabar/sales_data_analysis_visualization
data1 = pd.read_csv("https://raw.githubusercontent.com/devbabar/sales_data_analysis_visualization/master/WA_Sales_Products_2012-14.csv")
data1
data = data1.sample(n = 100)
px.scatter(data_frame = data)
px.scatter(data_frame = data,
           x = 'Revenue')
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity')
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type')
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line')
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity')
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity',
           hover_name = 'Retailer country')
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity',
           hover_name = 'Retailer country',
           custom_data = ['Year', 'Quarter'])
px.scatter(data_frame = data,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity',
           hover_name = 'Retailer country',
           custom_data = ['Year', 'Quarter'],
           text = 'Order method type')
# Removing the clutter for a better graph
# Removed text attribute and added height attribute.
px.scatter(data_frame = data,
           height = 1000,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity',
           hover_name = 'Retailer country',
           facet_row = 'Order method type')
px.scatter(data_frame = data,
           height = 600,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity',
           hover_name = 'Retailer country',
           facet_col = 'Order method type')
px.scatter(data_frame = data,
           height = 1000,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           symbol = 'Product line',
           size = 'Quantity',
           hover_name = 'Retailer country',
           facet_col = 'Order method type',
           facet_col_wrap = 3)
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 600,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           error_x='Quantity')
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 600,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           error_y='Quantity')
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 600,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           error_x='Quantity',
           error_x_minus='Gross margin')
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 1000,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Retailer type',
           error_y='Quantity',
           error_y_minus='Gross margin')
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Revenue',
           y = 'Quantity',
           range_x = [20000, 50000])
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Revenue',
           y = 'Quantity',
           range_y = [0, 5000])
# Removed few parameter to clean the clutter
#To understand animation we have group the data and then check
# The data is grouped so that animation parameter can be understood.
df_for_animation = data.groupby(['Retailer type', 'Year'])[['Revenue', 'Quantity']].agg('sum').reset_index()
fig1 = px.scatter(data_frame = df_for_animation, 
           x = 'Retailer type', 
           y = 'Revenue', 
           color = 'Retailer type', 
           animation_frame = 'Year',
           range_y = [0,600000])
fig1
fig2 = px.scatter(data_frame = df_for_animation, 
           x = 'Retailer type', 
           y = 'Revenue', 
           color = 'Retailer type', 
           animation_frame = 'Year',
           animation_group = 'Retailer type',
           range_y = [0,1000000])
fig2
px.scatter(data_frame = df_for_animation, 
           x = 'Retailer type', 
           y = 'Revenue',
           category_orders = {'Retailer type' : 
                              ['Equipment Rental Store', 'Direct Marketing', 
                               'Department Store', 'Warehouse Store', 
                               'Eyewear Store', 'Golf Shop', 'Outdoors Shop', 
                               'Sports Store']
                             }
          )
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Revenue',
           y = 'Quantity',
           labels = {'Revenue':'Changed_Name_Xaxis', 'Quantity': 'Changed_Name_Yaxis'})
# This is without the above paramter.
fig3 = px.scatter(data_frame = df_for_animation, 
           x = 'Retailer type', 
           y = 'Revenue', 
           color = 'Retailer type')
fig3
fig4 = px.scatter(data_frame = df_for_animation, 
           x = 'Retailer type', 
           y = 'Revenue', 
           color = 'Retailer type',
           color_discrete_sequence = ["red", "green", "blue", "goldenrod", "magenta", 'yellow', 'black', 'pink'])
fig4
fig5 = px.scatter(data_frame = df_for_animation, 
           x = 'Retailer type', 
           y = 'Revenue', 
           color = 'Retailer type',
           color_discrete_map = {'Equipment Rental Store':'green', 
                                      'Direct Marketing':'blue', 
                                      'Department Store':'goldenrod', 
                                      'Warehouse Store':'magenta', 
                                      'Eyewear Store':'yellow', 
                                      'Golf Shop':'red', 
                                      'Outdoors Shop':'black', 
                                      'Sports Store':'pink'
                                     }
                 )
fig5
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Revenue',
           color_continuous_scale = 'viridis')

# color scales - 
#https://plotly.com/python/builtin-colorscales/
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Revenue',
           range_color = [0,200000])

#This graph is same as above but the range of color on the left is now limited to 200k.
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Revenue',
           y = 'Quantity',
           color = 'Revenue',
           color_continuous_scale = 'viridis',
           color_continuous_midpoint = 150000)
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Retailer type',
           y = 'Revenue',
           symbol = 'Retailer type',
           symbol_sequence = ['circle','square-open','cross-open','triangle-ne-open','pentagon-dot','hourglass-open','diamond-x-open','cross-thin-open']
          )
# Removed few parameter to clean the clutter
px.scatter(data_frame = data,
           height = 500,
           x = 'Retailer type', 
           y = 'Revenue',
           symbol = 'Retailer type',
           symbol_map = {'Equipment Rental Store':'circle', 
                                      'Direct Marketing':'square-open', 
                                      'Department Store':'cross-open', 
                                      'Warehouse Store':'triangle-ne-open', 
                                      'Eyewear Store':'pentagon-dot', 
                                      'Golf Shop':'hourglass-open', 
                                      'Outdoors Shop':'diamond-x-open', 
                                      'Sports Store':'cross-thin-open'
                                     }
          )
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue', 
           color = 'Retailer type',
           opacity = .4)
px.scatter(data_frame = data, 
           x = 'Year', 
           y = 'Revenue',
           size = 'Quantity',
           title = 'Without Max Size')
px.scatter(data_frame = data, 
           x = 'Year', 
           y = 'Revenue',
           size = 'Quantity',
           size_max = 10,
           title = 'With Max Size')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           marginal_y = 'box')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           marginal_x = 'box')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           marginal_y = 'box',
           marginal_x = 'box')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           trendline = 'ols')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           color = 'Retailer type',
           trendline = 'ols')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           trendline = 'ols',
           trendline_color_override = 'red')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           log_x = True)
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           log_y = True)
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           render_mode = 'svg')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           title = 'This is a title')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           title = 'This is a title',
           template = 'plotly_dark')
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           width = 500)
px.scatter(data_frame = data, 
           x = 'Quantity', 
           y = 'Revenue',
           height = 300)