import pandas as pd  #pandas does things with matrixes

import numpy as np #used for sorting a matrix

import matplotlib.pyplot as plt #matplotlib is used for plotting data

import matplotlib.ticker as ticker #used for changing tick spacing

import datetime as dt #used for dates

import matplotlib.dates as mdates #used for dates, in a different way
allstock = pd.read_csv('../input/all_stocks_5yr.csv') #reads the file
allstock.columns #prints just the columns of the matrix
stocknames=allstock.Name.unique() #pulls all unique names from column 'name'

stocknames=np.sort(stocknames,kind='quicksort') #sorts them alphabetically

print(stocknames) #displays the matrix of the names
total = allstock.isnull().sum().sort_values(ascending=False) #counts all null cells in a row

percent = ((allstock.isnull().sum()/allstock.isnull().count()).sort_values(ascending=False)*100) #sees what percent of the data is null

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent']) #combines the two matrixies

missing_data #this displays the matrix
allstock = allstock.drop(allstock.loc[allstock['Volume'].isnull()].index) #drops rows with a null cell in the Volume column

allstock = allstock.drop(allstock.loc[allstock['Open'].isnull()].index) #drops rows with a null cell in the Open column
total = allstock.isnull().sum().sort_values(ascending=False)

percent = ((allstock.isnull().sum()/allstock.isnull().count()).sort_values(ascending=False)*100)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data
fig_size = plt.rcParams["figure.figsize"] #loads current figure size

print('old size:',fig_size) #prints the size

fig_size[0] = 15 #sets the X size to 15

fig_size[1] = 8 #sets the Y size to 8

plt.rcParams["figure.figsize"] = fig_size #sets this numbers to the new size

fig_size = plt.rcParams["figure.figsize"] #loads the figure size for checking

print ('new size:',fig_size) #prints the figure size
stock = 'AMZN' #edit the stock name to view other stocks. see the cell above with a list of stocknames

catagory = 'High' #edit this to change which value is plotted (see allstock.columns cell for options)

allstocksingle = allstock[allstock['Name'] == stock] #makes matrix with only the stock info



x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in allstocksingle['Date']] #convert date to something python understands

y = allstocksingle[catagory] #plots which ever catagory you entered above



plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y')) #display the date properly

plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60)) #x axis tick every 60 days

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(100)) # sets y axis tick spacing to 100



plt.plot(x,y) #plots the x and y

plt.grid(True) #turns on axis grid

plt.ylim(0) #sets the y axis min to zero

plt.xticks(rotation=90,fontsize = 10) #rotates the x axis ticks 90 degress and font size 10

plt.title(stock) #prints the title on the top

plt.ylabel('Stock Price For '+ catagory) #labels y axis

plt.xlabel('Date') #labels x axis



#plt.savefig(stock+catagory+'.png')
startdate = ('2013-01-01') #enter the start date here, it must be YYYY-MM-DD

enddate = ('2014-01-01') #enter the end date here, it must be YYYY-MM-DD



plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y')) #display the date

plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20)) #x axis tick every 20 days

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50)) # sets y axis tick spacing to 50



plt.plot(x,y) #plots the x and y

plt.grid(True)

plt.xlim(startdate,enddate) #this is the new line of code that sets the start and end limits on the x axis

plt.ylim(0, 500) #sets the y axis min to zero and y max to 500

plt.xticks(rotation=90,fontsize = 10) #rotates the x axis ticks 90 degress and font size 10

plt.title(stock+ '  '+ startdate + ' to ' + enddate) #prints the title on the top



plt.ylabel('Stock Price For '+ catagory) #labels y axis

plt.xlabel('Date') #labels x axis