import pandas as pd

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.plotting import figure, show, output_file
def get_portfolio(list_of_stocks):



    stocks = {}

    tickers = []

    

    for i in list_of_stocks:

    

        stocks[i] = pd.read_csv('../input/'+str(i)+'.csv', na_values = ['NaN','','nan'], keep_default_na = False)

        

        tickers.append(i)       

    

    portfolio = pd.concat(stocks.values(), axis=1, keys=tickers)



    portfolio.columns.names = ['Stock ticker', 'Stock info']



    return portfolio, stocks, tickers
def clean_portfolio(stocks):

        

    stocks_clean = {}

        

    #Any values with a value of less than half of the standard deviation is considered anomalous and will be replaced by the mean of the column.    

    def anomolous_result(series):

            

            for i in range(len(series)):



                if series.iloc[i] < (0.5*series.std()):

                    series.iloc[i] = series.mean() - series.iloc[i]

                else:

                    pass



            return series

        

    

    for i in tickers:

        

        #choose relevant columns

        stocks_clean[i] = stocks[i].drop(['Open','High','Low','Adj. close**'],axis=1).dropna()

        

        #reset column names

        stocks_clean[i].columns = ['date','close','volume']

        

        #format the columns

        stocks_clean[i]['date'] = pd.to_datetime(stocks_clean[i]['date'])

        stocks_clean[i]['close'] = pd.to_numeric(stocks_clean[i]['close'],errors='coerce')

        stocks_clean[i]['volume'] = stocks_clean[i]['volume'].apply(lambda x:x.replace(',',''))

        stocks_clean[i]['volume'] = stocks_clean[i]['volume'].astype('int32')

        

        #sort the stocks dataframes by the date column so that it is ascending

        stocks_clean[i].sort_values(by='date', inplace=True)

        

        #Addressing anomolous values

        stocks_clean[i]['close'] = anomolous_result(stocks_clean[i]['close'])

        

    stocks = stocks_clean

    

    return stocks
def volume(tickers):

    

    volume = pd.DataFrame(tickers['MKS']['date'])

    

    for i in stocks:

        

        volume[i] = tickers[i]['volume']



    volume['average volume'] = volume.mean(axis=1)



    return volume
def close_prices(stocks):

    

    closing_prices = pd.DataFrame(stocks['MKS']['date'])

    

    for i in tickers:

        

        closing_prices[i] = stocks[i]['close']

    

    return closing_prices
def daily_change(close_prices_table):

    

    daily_change_table = pd.DataFrame(close_prices_table['date'])

    

    for i in tickers:

       

        daily_change_table[i +' daily change'] = close_prices_table[i].diff()

   

    daily_change_table['average daily change'] = daily_change_table.mean(axis=1)

    

    #drop the first row since it will be full of null values

    daily_change_table.dropna(thresh=4, inplace=True)

    

    return daily_change_table
def daily_return(close_prices_table):

    

    daily_return_table = pd.DataFrame(close_prices_table['date'])



    for i in tickers:

        

        daily_return_table[i+' daily return'] = close_prices_table[i].pct_change()*100



    daily_return_table['average daily return'] = daily_return_table.mean(axis=1)

    

    #drop the first row because it will contain null values

    daily_return_table.dropna(thresh=4, inplace=True)

    

    return daily_return_table
def investment_return(close_prices_table):

    

    investment_return_table = pd.DataFrame(close_prices_table['date'])

    

    def return_from_purchase(x):

        return ((x-(close_prices_table[i].iloc[0]))/(close_prices_table[i].iloc[0]))*100



    for i in tickers:

        

        investment_return_table[i + ' investment return'] = close_prices_table[i].apply(lambda x: return_from_purchase(x))



    investment_return_table['average investment return'] = investment_return_table.mean(axis=1)

    

    #Drop the first row because it will just contain null values

    investment_return_table = investment_return_table[1::]

    

    return investment_return_table
def format_graph(dataframe,title,x_label,y_label):

   

    #Define the data source for the plots

    source = ColumnDataSource(data = dataframe)



    #Instantiate the figure for plotting

    fig = figure(plot_width = 1200, 

                              plot_height = 600, 

                              title = title, 

                              x_axis_label = x_label, 

                              y_axis_label = y_label,

                              x_axis_type="datetime",

                              tools = ['wheel_zoom','box_select','box_zoom','reset']

                              )



    fig.background_fill_color = "DeepSkyBlue"

    fig.background_fill_alpha = 0.05

    fig.xgrid.grid_line_color = "white"

    fig.ygrid.grid_line_color = "white"

    

    #Define a list containing each ticker so that we can iterate through each company's data

    lines = list(dataframe.drop('date',axis=1))

    colors = {tickers[0]:'black',tickers[1]:'green',tickers[2]:'orange',tickers[3]:'darkblue'}

    

    return source, fig, lines, colors
def plot_graph(source, figure, title, lines, colors):

    

    for i in lines:



            if i.split(' ')[0] in tickers:



                #plot the data onto the figure

                plot = figure.line(x= 'date',

                                   y= i,

                                   legend_label=i,

                                   line_width=0.8,

                                   line_color=colors[i.split(' ')[0]],

                                   alpha=0.8,

                                   source=source

                                   )

                

                #add the tools to the graph

                figure.add_tools(HoverTool(renderers = [plot],

                                           tooltips = [('Company', i),

                                                       ( 'date',   '@date{%F}' ),

                                                       ( str(title),  '@'+'{'+i+'}' ),

                                                      ],

                                           formatters = {

                                                        'date' : 'datetime',

                                                        str(title) : 'printf'

                                                        }

                                          )

                                )

 

            else:



                #plot the data onto the figure

                plot = figure.line(x= 'date',

                                   y= i,

                                   legend_label=i,

                                   line_width=3,

                                   line_color='red',

                                   alpha=1,

                                   source=source

                                  )



                #add the tools to the graph

                figure.add_tools(HoverTool(renderers = [plot],

                                                     tooltips = [('Company', i),

                                                                 ( 'date',   '@date{%F}' ),

                                                                 ( str(title),  '@'+'{'+i+'}' ),

                                                                ],

                                                     formatters = {

                                                                 'date' : 'datetime',

                                                                 str(title) : 'printf'

                                                                  },

                                                     mode='vline'

                                                  )

                                        )



    return figure
portfolio, stocks, tickers = get_portfolio(['MKS','MRW','SBRY','TSCO'])
portfolio.head(2)
stocks = clean_portfolio(stocks)
volume_table = volume(stocks)
volume_table.head(3)
close_prices_table = close_prices(stocks)

#Addressing missing values for may 19th

close_prices_table['MRW'].loc[36] = close_prices_table['MRW'].mean()

close_prices_table['TSCO'].loc[36] = close_prices_table['TSCO'].mean()
close_prices_table.head(3)
daily_change_table = daily_change(close_prices_table)
daily_change_table.head(3)
daily_return_table = daily_return(close_prices_table)
daily_return_table.head(3)
investment_return_table = investment_return(close_prices_table)
investment_return_table.head(3)
volume_source, volume_figure, volume_lines, volume_color = format_graph(dataframe = volume_table,

                                                                        title = 'volume',

                                                                        x_label = 'time',

                                                                        y_label = 'volume')
volume_plot = plot_graph(source = volume_source, 

                         figure = volume_figure, 

                         title = 'volume', 

                         lines = volume_lines, 

                         colors = volume_color)
show(volume_plot)
close_source, close_figure, close_lines, close_color = format_graph(dataframe = close_prices_table,

                                                                    title = 'closing price',

                                                                    x_label = 'time',

                                                                    y_label = 'closing price')
close_prices_plot = plot_graph(source = close_source, 

                               figure = close_figure, 

                               title = 'closing price (£)', 

                               lines = close_lines, 

                               colors = close_color)
show(close_prices_plot)
dc_source, dc_figure, dc_lines, dc_color = format_graph(dataframe = daily_change_table,

                                                        title = 'daily change',

                                                        x_label = 'time',

                                                        y_label = 'daily change')
daily_change_plot = plot_graph(source = dc_source, 

                               figure = dc_figure, 

                               title = 'daily change (£)', 

                               lines = dc_lines, 

                               colors = dc_color)
show(daily_change_plot)
dr_source, dr_figure, dr_lines, dr_color = format_graph(dataframe = daily_return_table,

                                                        title = 'daily return',

                                                        x_label = 'time',

                                                        y_label = 'daily return')
daily_return_plot = plot_graph(source = dr_source, 

                               figure = dr_figure, 

                               title = 'daily return (%)', 

                               lines = dr_lines, 

                               colors = dr_color)
show(daily_return_plot)
ir_source, ir_figure, ir_lines, ir_color = format_graph(dataframe = investment_return_table,

                                                        title = 'investment return',

                                                        x_label = 'time',

                                                        y_label = 'investment return')
investment_return_plot = plot_graph(source = ir_source, 

                                    figure = ir_figure, 

                                    title = 'investment return (%)', 

                                    lines = ir_lines, 

                                    colors = ir_color)
show(investment_return_plot)