# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show, curdoc, output_notebook, push_notebook
from bokeh.models.widgets import Dropdown, CheckboxGroup
from bokeh.layouts import row
from bokeh.palettes import inferno as palette
import itertools
output_notebook()
def update_plot(ticker_list):
    '''

    :param ticker_list: list of active ticker symbols in checkbox group
    :return: None
    '''

    curdoc().clear()
    plot = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='High Price', title='Stock Data Visualization')

    with open('a.txt', 'a') as f:
        f.write(f'{ticker_list}\n')
    pass

    # chose colors depending on number of stock symbols selected
    colors = itertools.cycle(palette(len(ticker_list)))

    for ticker, color in zip(ticker_list, colors):

        # Get the data corresponding to the ticker symbol
        stocks_ticker = all_stocks[all_stocks['Name'] == ticker]
        stocks_ticker['date'] = pd.to_datetime(stocks_ticker['date'])
        data_ticker = ColumnDataSource(stocks_ticker)

        plot.line(x='date', y='high', source=data_ticker, legend=ticker , color=color)
        # plot.circle(x='date', y='high', size=2, source=data_ticker, fill_color='white')

    layout1 = row([menu_dropdown, plot], sizing_mode='fixed', height=500, width=500)
    
    show(layout1,notebook_handle=True)

    curdoc().add_root(layout1)

#     for index, checkbox in cb_elements.items():
#         curdoc().add_root(checkbox)
#         checkbox.on_change('active', handle_checkbox_change)


    curdoc().title = f"{__name__}"



def handle_checkbox_change(val, old, new):
    '''

    :param val: val passed
    :param old: old value of val
    :param new: new value of val
    :return: None
    '''
    with open('a.txt', 'a') as f:
        f.write(f'{val}\t{old}\t{new}\n')
    pass

    # Find which elements are active
    active_elements = []
    with open('a.txt', 'a') as f:
        for key, val in cb_elements.items():
            if val.active == [0]:
                # f.write(f'{val}\t{val.active}\t{val.name}\t{val.labels}\n')
                active_elements.append(val)


    # Update the plot
    ticker_symbol_list = []
    for element in active_elements:
        name = element.labels[0]
        ticker_symbol_list.append(name.split('_')[1])
    update_plot(ticker_symbol_list)



def handle_menu_dropdown(attr, old, new):
    '''

    :param attr: attribure passed
    :param old: old value of the attribute
    :param new: new value of the attribute
    :return: None
    '''

    # Log events to file on server
    with open('a.txt', 'a') as f:
        f.write(f'{menu_dropdown.menu[1]}\t{old}\t{new}\t{type(new)}\n')

    search_func = lambda x: [y for y in x if y[1] == str(new)]
    val = search_func(menu_dropdown.menu)

    # Get the name and index
    name, index = val[0]
    
    # Bokeh is not working correctly, so displaying only one plot
    tl = [name]
    update_plot(tl)

#     # Create a checkbox dynamically
#     cb = None
#     if index not in cb_elements.keys():
#         cb = CheckboxGroup(labels=[f'Stock_{name}'], active=[1])
#         cb_elements[index] = cb
#         curdoc().add_root(cb)
#         cb.on_change('active', handle_checkbox_change)

if True:

    # Read the file containing the stocks data as csv
    filename = "all_stocks_5yr.csv"

    cb_elements = {}

    # Get the Ticker symbol
    all_stocks = pd.DataFrame(pd.read_csv('../input/'+filename))
    ticker_symbols = all_stocks['Name'].unique()
    ticker_symbols.sort()

    # Create a Dropdown to select the symbols
    ticker_menu = list( (ticker, str(index)) for index, ticker in enumerate(ticker_symbols))
    menu_dropdown = Dropdown(label='Stocks', menu=list(ticker_menu))

    # Add callback function
    menu_dropdown.on_change('value', handle_menu_dropdown)

    # Plot selected without selecting any symbol. Initial plot
    stocks_apple = all_stocks[all_stocks['Name'] == 'AAL']
    stocks_apple['date'] = pd.to_datetime(stocks_apple['date'])
    stocks_wdc = all_stocks[all_stocks['Name'] == 'WDC']
    stocks_wdc['date'] = pd.to_datetime(stocks_wdc['date'])

    # Plotting using dataframe
    # plot = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='High Price')
    # plot.line(x=stocks_apple['date'], y=stocks_apple['high'], color='blue')
    # plot.line(x=stocks_wdc['date'], y=stocks_wdc['high'], color='red')


    # plotting using ColumnDataSource
    dataapple = ColumnDataSource(stocks_apple)
    datawdc = ColumnDataSource(stocks_wdc)

    plot = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='High Price', title='Stock Data Visualization')
    plot.line(x='date', y='high', source=dataapple, color='red')
    plot.circle(x='date', y='high', size=2, source=dataapple, fill_color='white')

    plot.line(x='date', y='high', source=datawdc, color='green')
    plot.circle(x='date', y='high', size=2, source=datawdc, fill_color='black')

    layout1 = row([menu_dropdown, plot], sizing_mode='fixed', height=130, width=350)

    # Add the plot to the current doc
    curdoc().add_root(layout1)
    curdoc().title = f"{__name__}"
show(layout1, notebook_handle=True)


