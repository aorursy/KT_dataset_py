!pip install yfinance
import yfinance as yf



import pandas as pd

import plotly.graph_objects as go
# Example of how yfinance works



# Request stocks data for Microsoft (MSFT)

MSFT = yf.Ticker("MSFT")

df_MSFT = MSFT.history(period="max")



# Display the dataset

df_MSFT
# Request stocks data for Apple (AAPL)

AAPL = yf.Ticker("AAPL")

df_AAPL = AAPL.history(period="max")



# Request stocks data for Amazon (AMZN)

AMZN = yf.Ticker("AMZN")

df_AMZN = AMZN.history(period="max")



# Request stocks data for Google (GOOGL)

GOOGL = yf.Ticker("GOOGL")

df_GOOGL = GOOGL.history(period="max")
df_stocks = pd.DataFrame({

    'MSFT': df_MSFT['High'],

    'AAPL': df_AAPL['High'],

    'AMZN': df_AMZN['High'],

    'GOOGL': df_GOOGL['High'],

})



# manually create a dataset of stocks at their daily High

df_stocks
# How to change plot data using dropdowns

#

# This example shows how to manually add traces

# to the plot and configure the dropdown to only

# show the specific traces you allow.



fig = go.Figure()



for column in df_stocks.columns.to_list():

    fig.add_trace(

        go.Scatter(

            x = df_stocks.index,

            y = df_stocks[column],

            name = column

        )

    )

    

fig.update_layout(

    updatemenus=[go.layout.Updatemenu(

        active=0,

        buttons=list(

            [dict(label = 'All',

                  method = 'update',

                  args = [{'visible': [True, True, True, True]},

                          {'title': 'All',

                           'showlegend':True}]),

             dict(label = 'MSFT',

                  method = 'update',

                  args = [{'visible': [True, False, False, False]}, # the index of True aligns with the indices of plot traces

                          {'title': 'MSFT',

                           'showlegend':True}]),

             dict(label = 'AAPL',

                  method = 'update',

                  args = [{'visible': [False, True, False, False]},

                          {'title': 'AAPL',

                           'showlegend':True}]),

             dict(label = 'AMZN',

                  method = 'update',

                  args = [{'visible': [False, False, True, False]},

                          {'title': 'AMZN',

                           'showlegend':True}]),

             dict(label = 'GOOGL',

                  method = 'update',

                  args = [{'visible': [False, False, False, True]},

                          {'title': 'GOOGL',

                           'showlegend':True}]),

            ])

        )

    ])



fig.show()
# Configure functions to automate the data collection process



# getStocks requires three variables:

# - stocks is a list of strings which are the code for the stock

# - history is timeframe of how much of the stock data is desired

# - attribute is the attribute of the stock 

def getStocks(stocks, history, attribute):

    return pd.DataFrame({stock:yf.Ticker(stock).history(period=history)[attribute] for stock in stocks})



# multi_plot requires two variables:

# - df is a dataframe with stocks as columns and rows as date of the stock price

# - addAll is to have a dropdown button to display all stocks at once

def multi_plot(df, addAll = True):

    fig = go.Figure()



    for column in df.columns.to_list():

        fig.add_trace(

            go.Scatter(

                x = df.index,

                y = df[column],

                name = column

            )

        )



    button_all = dict(label = 'All',

                      method = 'update',

                      args = [{'visible': df.columns.isin(df.columns),

                               'title': 'All',

                               'showlegend':True}])



    def create_layout_button(column):

        return dict(label = column,

                    method = 'update',

                    args = [{'visible': df.columns.isin([column]),

                             'title': column,

                             'showlegend': True}])



    fig.update_layout(

        updatemenus=[go.layout.Updatemenu(

            active = 0,

            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))

            )

        ])

    

    fig.show()
# How to change plot data using dropdowns

#

# This example shows how to automatically add traces

# to the plot and automatically configure the dropdown

# to include all columns of the dataframe



# The beauty of this example is that all you need to do

# is to change the list of stock codes below and the plot

# will adjust accordingly



stocks = ['MSFT', # Microsoft

          'AAPL', # Apple

          'AMZN', # Amazon

          'GOOGL' # Google Alphabet

         ]



df_stocks = getStocks(stocks, 'max', 'High')



multi_plot(df_stocks)
# How to change plot data using dropdowns

#

# This example shows how to automatically add traces

# to the plot and automatically configure the dropdown

# to include all columns of the dataframe



# The beauty of this example is that all you need to do

# is to change the list of stock codes below and the plot

# will adjust accordingly



stocks = ['HMC', # Honda

          'TM', # Toyota Motor Corporation

          'GM', # General Motors

          'F', # FORD

          'TSLA' # Tesla

         ]



df_stocks = getStocks(stocks, 'max', 'High')



multi_plot(df_stocks)