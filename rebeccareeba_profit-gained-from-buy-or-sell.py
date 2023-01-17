# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Tools

from datetime import datetime as dt

from datetime import timedelta



#IPython manager

from IPython.display import display



#Graphs

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

df = pd.read_csv("../input/prices.csv", parse_dates=['date'])

df.head()

# Any results you write to the current directory are saved as output.
ticker_list = df['symbol'].unique() 

print(ticker_list)
date_list = df['date'].unique() 

date_list
df.sample(10)
year_list = df["date"].dt.year

print(year_list.unique())
#Create a new df for Apple tickers

df_AMZN = df[df['symbol'] == 'AMZN']

#Sort the dataframe by date

df_AMZN.sort_values(by=['date'], inplace=True, ascending= True)

# reset index to have sequential numbering

df_AMZN = df_AMZN.reset_index(drop=True)



df_AMZN.head()
# Create a new column to show the change in percentage between open and close

df_AMZN['change'] = (df_AMZN['close'] / df_AMZN['open']-1).fillna(0)

df_AMZN.head()
print(df_AMZN.index.max())
print(df_AMZN.loc[df_AMZN.index.max()-261]['open'])
# create a list of probable % change between open and close

rise = [.0009,.001,.002,.003,.004,.005,.008,.01]

fall = [-.01,-.011,-.012,-.014,-.015,-.016,-.02,-.04]



d = 251 # Nummber of days to invest (number of working days in year)

initial_amt = 1000 # initial amount to invest in dollars

total_profit = np.zeros([8, 8],dtype = float)
df_AMZN.shape
# define the starting date to invest from

start_index = (df_AMZN.index.max()-d)

start_index
#Convert dollars to invest into shares based on open price of the first day

#the 2016 data starts on 4 Jan, we find the open value on that date

start_open = df_AMZN.loc[start_index]['open']

# use the first days open value to deduce number of shares

shares = initial_amt/start_open



print(shares)
df_AMZN.reset_index(inplace=True)
x=0 # iterator for the rise loop

#loops through posible percentage increase in stock price

for b in rise: 

    

    y=0#iterator for the fall loop

    #loops through posible percentage decrease in stock price

    for s in fall:

        

        available_funds = 0.0

        temp_shares= shares

        #loop throught using index

        for i in range(start_index,df_AMZN.index.max()):

            

            daily_change = df_AMZN.loc[i]['change']

            share_price = df_AMZN.loc[i]['open']

            

            #Sell shares if price ffalls below certain percentage

            if daily_change <=s and temp_shares>0.0:

                #liquidate shares 

                available_funds = temp_shares*share_price

                temp_shares = 0

            #Buys shares if price rises above certain percentage

            elif daily_change >= b and available_funds>0:

                temp_shares = available_funds/share_price

                available_funds = 0

            else:

                continue

        

        #Total funds generated at the end of the cycle

        if temp_shares > 0:

            final_fund = temp_shares*share_price

        else:

            final_fund = available_funds

        

        # Calculate profit

        #print(final_fund)

        profit = final_fund-initial_amt

        total_profit[y,x] = profit

        y+=1

    x+=1

print(total_profit)      
max_profit = np.amax(total_profit)

# find the index of the highest profit

result = np.where(total_profit == max_profit)



final_rise = rise[int(result[1])]

final_fall = fall[int(result[0])]

print(max_profit, final_rise, final_fall)
print('The maximum profit we made Amazon stock investment of $1000 is ${:,.2f}'.format(max_profit),'/n')

print('The positive change percentage delimiter used ',final_rise)

print('The negative change percentage delimiter used ',final_fall)