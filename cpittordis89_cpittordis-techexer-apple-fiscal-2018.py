# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statsmodels.api as sm # import statsmodels

#from sklearn import linear_model ## Using Sci-kitlearn for regression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Input new Apple Inc's Fisca 2017 info from link: https://www.apple.com/uk/newsroom/2017/11/apple-reports-fourth-quarter-results



## Note: the data is only for AAPL, containing only 6 rows, not all elements are filled out.

dataFund = pd.read_csv('../input/funds-aapleditcsv/Funds_AAPL.csv', 'wb' , delimiter=',', encoding="utf-8-sig", engine='python')

index = dataFund['index']

Ticker= dataFund['Ticker Symbol']

#print (Ticker[0]) # Python 3 print args = print(what you want to print out)

period = dataFund['Period Ending']

Accounts=dataFund['Accounts Payable']

TotRevenue=dataFund['Total Revenue']



## Located Apple Inc data from /input/fundamentals.csv, using:

'''

dataFund = pd.read_csv('../input/nyse/fundamentals.csv', 'wb' , delimiter=',', encoding="utf-8-sig", engine='python')

index = dataFund['Unnamed: 0']

Ticker= dataFund['Ticker Symbol']

period = dataFund['Period Ending']

Accounts=dataFund['Accounts Payable']

TotRevenue=dataFund['Total Revenue']



for i in range(len(index)):

    if Ticker[i]==AAPL:

        print( period[i], Total Revenue[i]..etc )'''



###

print('Apple Inc Total Revenue 2013-09-28 -> 2017-09-24, links:\n https://www.apple.com/uk/newsroom/2017/11/apple-reports-fourth-quarter-results/ \n https://www.apple.com/newsroom/pdfs/fy17-q4/Q4FY17ConsolidatedFinancialStatements.pdf \n https://www.apple.com/newsroom/pdfs/fy17-q4/Q4FY17DataSummary.pdf')

print('Index \t Fiscal year\t\t \tApple Ticker\t\tTotal Revenue[Billons]\tPercentage change year')

for i in range(len(index)):

    j=i-1

    if j==-1:#  period[i] >= '2011-\n-\n' and Ticker[i] == 'AAPL':## used to search for Apple inc's data

        j=0

        i=0

        percen =100- (((TotRevenue[j]/TotRevenue[i]))*100)

        #percen+=1

        print('{}\t {}--{} \t\t {} \t\t {} \t\t {:.3f}%'.format(index[i],period[j],period[i],Ticker[i],(TotRevenue[i]/1e9), (percen)))

    else:

        percen =100- (((TotRevenue[j]/TotRevenue[i]))*100)

        print('{}\t {}--{} \t\t {} \t\t {} \t\t {:.3f}%'.format(index[i],period[j],period[i],Ticker[i],(TotRevenue[i]/1e9), (percen)))

        plt.plot(period[i],(TotRevenue[i]/1e9),'r.')

plt.ylabel('Total Revenue [Billions]') 

plt.xlabel('Fiscal year period')

plt.show()



X = index #dataFund["Period Ending"] ## X usually means our input variables (or independent variables)

y = TotRevenue #target["Total Revenue"] ## Y usually means our output/dependent variable

X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model



# Note the difference in argument order

model = sm.OLS(y, X).fit() ## sm.OLS(output, input)

predictions = model.predict(X)



# Print out the statistics

model.summary()
## using the OLS Regression results from the above table

def Apple_Fiscal(a,c,m):

    b = (a*m) + c

    return b



a = 5 ## represents 2017-2018 fiscal year

m = 1.495e10 ## coeff from above OLS regression table

c = 1.766e11 ## const/intercept from above OLS regression table

Perc = 100-((TotRevenue[4]/Apple_Fiscal(a,c,m))*100) 

print('{} {} Billion '.format('Apple Inc Predicted Fiscal 2017-2018 =', Apple_Fiscal(a,c,m)/1e9))

print('{} {:.3f}%'.format('With percentage year change = ', Perc))



print('\nThe results can be updated with the Apple inc Third Quarter 2018 Results, given by this link: \nhttps://www.apple.com/uk/newsroom/2018/07/apple-reports-third-quarter-results/')