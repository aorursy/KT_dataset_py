import numpy as np

import pandas as pd

import imageio

import datetime

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
ZHVI = pd.read_csv("../input/median-housing-price-us/Affordability_ChainedZHVI_2017Q2.csv")

CPI = pd.read_csv("../input/consumer-price-index-usa-all-items/USACPIALLMINMEI.csv")

Income = pd.read_csv("../input/median-housing-price-us/Affordability_Income_2017Q2.csv")
ZHVI.head()
Income.head()
CPI.head()
# melt ZHVI to create Date and ZHVI columns

ZHVI = pd.melt( ZHVI, id_vars=['RegionID', 'RegionName', 'SizeRank'], value_name='ZHVI', var_name = 'Date')

# melt Income to create Date and Income columns

Income = pd.melt( Income, id_vars=['RegionID', 'RegionName', 'SizeRank'], value_name='Income', var_name = 'Date')

#merge ZHVI and Income dataframes on ['RegionID','RegionName','SizeRank','Date'] columns

ZHVI = ZHVI.merge(Income, how='outer', on=['RegionID','RegionName','SizeRank','Date'])

#rename CPI columns

CPI.columns = ['Date', 'CPI']

#change CPI Date column values from string to datetime object 

CPI.Date = pd.to_datetime(CPI.Date, format="%Y-%m")

#change ZHVI Date column values from string to datetime object

ZHVI.Date = pd.to_datetime(ZHVI.Date, format="%Y-%m")

#merge ZHVI and CPI on Date columns

ZHVI = ZHVI.merge(CPI, how='inner', on=['Date'])

#set CPI index to Date

CPI.set_index('Date', inplace= True)

#calculate average CPI for 2019

base = np.mean(CPI.loc['2019'])[0]

#adjust all income values to the average 2019 dollar value

ZHVI['Income_A'] = ZHVI.Income*base/ZHVI.CPI

#adjust ZHVI values to the average 2019 dollar value

ZHVI['ZHVI_A'] = ZHVI.ZHVI*base/ZHVI.CPI

#set ZHVI index to Date

ZHVI.set_index('Date', inplace = True)


#function to calculate regression line parameters which will be used in resulting plots

def reg(X,y):

    regr = LinearRegression()

    regr.fit(X,y)

    a= regr.coef_

    b= regr.intercept_

    r= regr.score(X,y)

    return a,b, r



#function for creating a plot which will become a single frame of the resulting gif 

def plot_gif(date, data):

    #create subplot

    fig, (ax1) = plt.subplots(nrows =1, ncols =1, figsize=(10, 10), squeeze = True)

    

    #set x to median income by region and y to ZHVI by region fro the given month adjusted to the value of the 2019 dollar

    x= data[date]['Income_A']

    y= data[date]['ZHVI_A']

    

    #set US_x to median US income and US_y to total US ZHVI for the given month

    US_x = data[date].loc[data[date].RegionName == 'United States'].Income_A

    US_y = data[date].loc[data[date].RegionName == 'United States'].ZHVI_A



   #calculate regression line parameters for x and y

    a,b,r = reg(x.values.reshape(-1, 1),y.values.reshape(-1, 1))

    a= a.squeeze()

    b= b.squeeze()

    

    #set plot style

    plt.style.use('seaborn-whitegrid')

    #plot x and y

    plt.scatter(x,y, label = 'Adjusted ZHVI by Locality (Top 100)')

    #plot US_x and US_y in red

    plt.scatter(US_x, US_y, c='r', label = 'Adjusted ZHVI Total US')

    #set y axis limits

    ax1.set_ylim(0,1000000)

    #set x axis limits

    ax1.set_xlim(20000,100000)

    #set axis labels and plot title

    ax1.set_ylabel('ZHVI Adjusted to 2019 Dollars', fontsize = 'large')

    ax1.set_xlabel('Median Income Adjusted to 2019 Dollars', fontsize = 'large')

    ax1.set_title('Regional ZHVI vs. Median Income: '+ date )

    #create regression line equation string

    reg_equation = 'y= %fx + %f \nR-squared: %f' % (a,b,r)

    #annotate plot with regression line equation and r_squared value 

    plt.text(x= 21000, y=940000,s=reg_equation)

   

    #plot regression line

    plt.plot([20000,100000], [a*20000+b, a*100000+b])

    

    #format ticks to include $ sign

    formatter =ticker.FormatStrFormatter('$%d')

    ax1.yaxis.set_major_formatter(formatter)

    ax1.xaxis.set_major_formatter(formatter)

    ax1.legend(loc='upper right')

    

    # Used to return the plot as an image array

    fig.canvas.draw()       

    # draw the canvas, cache the renderer

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    

    return image



#limit data to the 100 largest regions within the US

data = ZHVI.loc[ZHVI.SizeRank<100].dropna()



#create gif by passing list of plots and desired fps value into function

imageio.mimsave('./bubble.gif', [plot_gif(date,data) for date in pd.unique(data.index.strftime('%Y-%m'))], fps=20)
