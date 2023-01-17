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
data = pd.read_csv('/kaggle/input/cc-transactions/cc_transactions.csv')

data.head()
class CreditCardReport:

        def __init__(self,user,filename):

            self.username = user

            self.file = pd.read_csv(filename)

        def print_greeting(self):

            print('Hello, '+ self.username +'!')

            print('Thank you for being a valued cardholder since '+ str(self.file.year.sort_values()[0])+'.')

        def print_most_recent(self):

            #sort_values() by list [year, month ,day]

            print(self.file.sort_index(ascending = False).head(5))

        def plot_spending_by_category(self):

            self.file.groupby('category').amount.sum().plot.pie()

        def plot_monthly_spending(self):

#           self.file.set_index(['year','month']).amount.plot.hist()

            self.file.groupby(['year','month']).amount.sum().plot.bar()

        def print_top_month(self):

#             most = self.file[self.file.amount==self.file.amount.max()].iloc[0]

            highest = self.file.groupby(['year','month']).amount.sum()

            most = highest[highest == highest.max()]

            most = most.reset_index().loc[0]

            print('You spent the most in ' + str(int(most.year))+'/'+str(int(most.month)) +' ('+str(most.amount)+').')

        def print_top_merchants(self):

            k = self.file.groupby('merchant').size().sort_values(ascending = False).head(3).index.tolist()

#           k = self.file.merchant.value_counts().sort_values(ascending = False).head(3).index.tolist() 

            for i in range(len(k)):

                print(str(i+1)+'. '+k[i])

        
# Quiz 4 Problem 3: Credit card report generator - main program

# from helper import CreditCardReport # create credit card report object

ccr = CreditCardReport('Alice', '/kaggle/input/cc-transactions/cc_transactions.csv')

# print greeting message with name of client and start year of transactions

ccr.print_greeting()
# print 5 most recent transactions, with the most recent ones first

print('\n* Recent transactions:') 

ccr.print_most_recent()
# plot pie chart of total spending by category

print('\n* Spending by merchant category:') 

ccr.plot_spending_by_category()
# plot bar chart of total spending for each month

print('\n* Monthly spending:') 

ccr.plot_monthly_spending() 

ccr.print_top_month()
# print names of top 3 merchants with the most number of transactions, indescending order

print('\n* Most frequently visited merchants:') 

ccr.print_top_merchants()