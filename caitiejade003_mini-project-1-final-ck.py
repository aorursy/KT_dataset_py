# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing my key for AlphaStocks

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

StockData_Key = user_secrets.get_secret("AlphaStocks")

###TEST 1 - Looking at all the data I can get, and trying to get a particular piece of data for a stock on a given day



import urllib.error, urllib.parse, urllib.request, json



def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib2.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



def daily_stock(symbol = "MSFT"):  #function with the stock symbol as a parameter

    # https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo

    key = StockData_Key  #smaller variable for the api key

    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+symbol+"&"+"apikey="+key  #sample url

    return safeGet(url)





data = json.load(daily_stock())  #loading the data from the api into a json format





#checking that it works

#print(data)

#print(data['Meta Data'])

#print(data["Time Series (Daily)"]["2020-02-21"])

print(data["Time Series (Daily)"]["2020-02-21"]["1. open"])

print(data["Time Series (Daily)"]["2020-02-21"]["4. close"])



#day_open = data["Time Series (Daily)"]["2020-02-21"]["1. open"]

#print(day_open)

###TEST 2 Create a formula to find the % change of a stock on a particular day by comparing the open and close prices



day_open = float(data["Time Series (Daily)"]["2020-02-21"]["1. open"]) #getting the open value for a particular day and turning it into a float rather than string

print("Open: " + str(day_open))



day_close = float(data["Time Series (Daily)"]["2020-02-21"]["4. close"]) #getting the close value for a particular day and turning it into a float rather than string



print("Close: " + str(day_close))



day_change = (day_close - day_open) / day_open #formula for change of price





print("Today's change: " + str(day_change))

print(format(day_change, ".2%"))  #turning it into a percent format
### Test 3 putting all the stock data gathering steps together



date_list_test = ['2020-02-21', '2020-02-20', '2020-02-19'] #made a list of dates so I could look at more than one at a time

stock_list = ["MSFT", "AAPL"] #creating my list of stocks to get data for



def daily_stock(symbol):

    # https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo

    key = StockData_Key

    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="+symbol+"&"+"apikey="+key

    return safeGet(url)  #getting safe url using the function above

    



for stock in stock_list:   #iterate through the list of stocks

        data = json.load(daily_stock(stock))  #grabbing the data for each stock symbol in my stock_list

        for day in date_list_test: #then look at each date 

            day_open = float(data["Time Series (Daily)"][day]["1. open"])  #finding the open price

            day_close = float(data["Time Series (Daily)"][day]["4. close"]) #finding the close price

            day_change = (day_close - day_open) / day_open #calculating the change

            print("On date %s " %day + "the %s stock changed " %stock + format(day_change, ".2%"))   
#Test 4 - making a csv file with this data



import os



### Add your code here

import csv



date_list_test2 = ['2020-02-21', '2020-02-20', '2020-02-19', '2020-02-18'] #looking at more dates so I could look at more than one at a time



with open('stockdata.csv', mode='w') as stock_file:  #open a new file in write mode and assign it to stock_file

    stock_writer = csv.writer(stock_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #assign parameters to the csv writer method, found this online so not sure what the purpose quotechar/quoting parts are

    stock_writer.writerow(['Date', 'Symbol', 'Percent Change'])             #write the first row to be a header

    for stock in stock_list:   #iterate through the list of stocks

        data = json.load(daily_stock(stock))  #grabbing the data for each stock symbol in my stock_list

        for day in date_list_test2: #then look at each date 

            day_open = float(data["Time Series (Daily)"][day]["1. open"])  #finding the open price

            day_close = float(data["Time Series (Daily)"][day]["4. close"]) #finding the close price

            day_change = (day_close - day_open) / day_open #calculating the change

            stock_writer.writerow([day, stock, format(day_change, ".2%")])                  #use writer and writerow method to add each key and it's value

    





### This will print out the list of files in your /working directory to confirm you wrote the file.





for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from datetime import timedelta, date #importing datetime functions



date_list_full = [] #making an empty list



def daterange(date1, date2):  #creating a function that takes two dates

    for n in range(int ((date2 - date1).days)+1):  #the formula finds the number of days in the date range, then we iterate through each of those numbers

        yield date1 + timedelta(n) #increments the date by one each time



start_dt = date(2020, 1, 26)  #assigns the start date

end_dt = date(2020, 2, 24)    #assigns the end date

for dt in daterange(start_dt, end_dt):

    date_list_full.append(dt.strftime("%Y-%m-%d")) #adds the date to the datelist

    

print(date_list_full) #checking    
#This code block will create my final csv of stock data

#sometimes this doesn't work the first time - KeyError: 'Time Series (Daily)' - but that key does exist and rerunning the module usually works



import os

date_list_nowknd = ['2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', 

                    '2020-02-07', '2020-02-10', '2020-02-11', '2020-02-12', '2020-02-13', '2020-02-14', '2020-02-18', '2020-02-19', '2020-02-20', '2020-02-21', '2020-02-24'] # dates sans weekends & holiday



#making this a dictionary instead so I can reference both the stock symbol and the company name

#stock_list = ["MSFT", "AAPL", "AMZN"] #creating my list of 3 stocks to get data for



stock_list = {"MSFT" : "Microsoft", "AAPL" : "Apple", "AMZN" : "Amazon"} #creating my list of 3 stocks to get data for



#csv library

import csv



with open('stockdata.csv', mode='w') as stock_file:  #open a new file in write mode and assign it to stock_file

    stock_writer = csv.writer(stock_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #assign parameters to the csv writer method, found this online so not sure what the purpose quotechar/quoting parts are

    stock_writer.writerow(['Date', 'Term', 'Percent Change'])             #write the first row to be a header

    for stock in stock_list.keys():   #iterate through the list of stocks

        data = json.load(daily_stock(stock))  #grabbing the data for each stock symbol in my stock_list

        #write the weekend dates

        stock_writer.writerow(['2020-01-26', stock_list[stock], 0.0])    

        stock_writer.writerow(['2020-02-01', stock_list[stock], 0.0])             

        stock_writer.writerow(['2020-02-02', stock_list[stock], 0.0]) 

        stock_writer.writerow(['2020-02-08', stock_list[stock], 0.0])            

        stock_writer.writerow(['2020-02-09', stock_list[stock], 0.0]) 

        stock_writer.writerow(['2020-02-15', stock_list[stock], 0.0])             

        stock_writer.writerow(['2020-02-16', stock_list[stock], 0.0]) 

        stock_writer.writerow(['2020-02-17', stock_list[stock], 0.0]) #holiday

        stock_writer.writerow(['2020-02-22', stock_list[stock], 0.0])

        stock_writer.writerow(['2020-02-23', stock_list[stock], 0.0])

        for day in date_list_nowknd: #then look at each date that's not a weekend

            day_open = float(data['Time Series (Daily)'][day]["1. open"])  #finding the open price

            day_close = float(data['Time Series (Daily)'][day]["4. close"]) #finding the close price

            day_change = ((day_close - day_open) / day_open) * 100 #changed this to find percent by multiplying by 100

            stock_writer.writerow([day, stock_list[stock], "{0:.2f}".format(day_change)])                  #use writer and writerow method to add each key and it's value, changed the way I formatted it to remove % symbol

    





### This will print out the list of files in your /working directory to confirm you wrote the file.





for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#I've downloaded my working file, and uploaded it as input. I am going to use the uploaded version for the rest of this.

#making a data frame from my csv file





stock_df = pd.read_csv('/kaggle/input/stock-data/stockdata_final2.csv')



print (stock_df.head(20))  #checking it out

print(stock_df.dtypes.value_counts())



#stock_df.plot(x='Date', figsize=(12,8))

stock_df.plot()
stock_df2 = stock_df.pivot(index='Date', columns='Term', values='Percent Change') #pivoting my data so each cell has a value and the stock categories are columns

print(stock_df2.head())
#Here is my first visualization : Comparing the change of tech stock prices over time. 

#Finding 1: These three stocks seem to trend together most of the time. 

#Finding 2: Microsoft has been the most volotile over the last month in terms of %change and it looks like Apple has been the most steady.

stock_df2.plot(figsize=(16,8))
#importing the API key for the news

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

News_Data = user_secrets.get_secret("NewsAPI")



#importing the library after installing

from newsapi import NewsApiClient



# Init with my API key

newsapi = NewsApiClient(api_key=News_Data)



#sample provided

#all_articles = newsapi.get_everything(q='bitcoin',

                                     # sources='bbc-news,the-verge',

                                     # domains='bbc.co.uk,techcrunch.com',

                                     # from_param='2020-02-01',

                                     # to='2020-02-15',

                                     # language='en',

                                     # sort_by='relevancy',

                                     # page=2)

#print("There were %d articles mentioning Bitcoin from BBC News and the Verge 2/1/20-2/15/20." %all_articles['totalResults'])



#looking at the search term "Apple" for a single day

all_articles = newsapi.get_everything(q='Apple',

                                      from_param='2020-02-01',

                                      to='2020-02-01',

                                      language='en',

                                      sort_by='relevancy')



print("There were %d articles mentioning Apple from all news sources on 2/1/20." %all_articles['totalResults'])



#I am going to make another csv, this time with the news data



search_terms = ['Microsoft', 'Apple', 'Amazon']



with open('newsdata.csv', mode='w') as news_file:  #open a new file in write mode and assign it to stock_file

    news_writer = csv.writer(news_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #assign parameters to the csv writer method, found this online so not sure what the purpose quotechar/quoting parts are

    news_writer.writerow(['Date', 'Term', 'Number of articles'])             #write the first row to be a header

    for term in search_terms:   #iterate through the list of search terms

        companyname = term #setting query to each item in the search term list     

        for date in date_list_full:

            all_articles = newsapi.get_everything(q=companyname,                                   

                                          from_param=date,

                                          to=date,

                                          language='en',

                                          sort_by='relevancy',)

            news_writer.writerow([date, companyname, all_articles['totalResults']])                  #use writer and writerow method to add each

    





### This will print out the list of files in your /working directory to confirm you wrote the file.





for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#downloading the data to reupload it, and will use uploaded csv that for my analysis

#creating a dataframe from the csv and seeing how it looks





news_df = pd.read_csv('/kaggle/input/news-data-final/newsdata_allsources.csv') #need to change this



print (news_df.head())  #checking it out

print(news_df.dtypes.value_counts())
#pivoting df



news_df2 = news_df.pivot(index='Date', columns='Term', values='Number of articles') #pivoting my data so each cell has a value and the terms are columns

print(news_df2.head())
# Visualization #2: For each day, see which company was mentioned the most

#Here is a barchart showing the number of articles for each term by day. 

#Findings: Looks like Amazon is most commonly referenced followed by Apple and Microsoft is last. 

news_df2.plot(kind = 'bar', figsize=(16,8))
#I am going to make another csv, this time with the news data from specific sources relevant to technology



with open('newsdata_techsources.csv', mode='w') as news_file:  #open a new file in write mode and assign it to stock_file

    news_writer = csv.writer(news_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #assign parameters to the csv writer method, found this online so not sure what the purpose quotechar/quoting parts are

    news_writer.writerow(['Date', 'Term', 'Number of articles'])             #write the first row to be a header

    for term in search_terms:   #iterate through the list of search terms

        companyname = term #setting query to each item in the search term list     

        for date in date_list_full:

            all_articles = newsapi.get_everything(q=companyname,

                                          sources='engadget, hacker-news, techcrunch, techradar, the-verge, wired',        

                                          from_param=date,

                                          to=date,

                                          language='en',

                                          sort_by='relevancy',)

            news_writer.writerow([date, companyname, all_articles['totalResults']])                  #use writer and writerow method to add each

    





### This will print out the list of files in your /working directory to confirm you wrote the file.





for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#creating a dataframe from the csv and seeing how it looks (using uploaded data)



technews_df = pd.read_csv('/kaggle/input/newstech/newsdata_techsources.csv') #normally use, /kaggle/working/newsdata_techsources.csv, but if amount of calls to the newsapi is exceeded, use /kaggle/input/newsdata_techsources_incase.csv



print (technews_df.head())  #checking it out



#pivoting df



technews_df2 = technews_df.pivot(index='Date', columns='Term', values='Number of articles') #pivoting my data so each cell has a value and the terms are columns

print(technews_df2.head())





#Barchart showing the number of articles for each term by day from technology-orientated sources. 

#Findings: In this chart we see Amazon and Apple competing for the top spot and Microsoft closer behind.

technews_df2.plot(kind = 'bar', figsize=(16,8))
#To combine these two I merged using both the date and term columns



technews_stocks = pd.merge(technews_df, stock_df, how='left', on=['Term','Date'])



technews_stocks.head(20)
#Visualization #3 - Comparing the number of articles with the change in the stock price with a scatter plot

technews_stocks[['Number of articles', 'Percent Change']].plot(kind='scatter', x='Number of articles', y='Percent Change')



#Findings: I can't currently see much in the way of correlation, but I want to clean this up some more
#First to clean up, I'll remove the rows with 0





# Get names of indexes for which column Percent Change has value 0

indexNames = technews_stocks[ technews_stocks['Percent Change'] == 0 ].index

 

# Delete these row indexes from dataFrame

technews_stocks.drop(indexNames , inplace=True)



technews_stocks.head(10)
#New Scatter plot, by removing the 0 values and adding a trend line, I see a very slight negative slope. So in the last month, it seems fewer articles about a company led to slightly better stock performance. 

#I guess no news is good news!



import numpy as np

import matplotlib.pyplot as plt



x = technews_stocks['Number of articles']

y = technews_stocks['Percent Change']



coef = np.polyfit(x,y,1)

poly1d_fn = np.poly1d(coef) 

# poly1d_fn is now a function which takes in x and returns an estimate for y



plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k')

plt.xlabel("Number of Articles")

plt.ylabel("% Change in Stock Value")
