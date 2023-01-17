# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas_datareader

import datetime

import matplotlib.pylab as plt

import seaborn as sns

from matplotlib.pylab import style

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# I made adjustment of the sample code and made marks in somewhere different from the original code.

# I delete the unnecessary code and graphs.

#!pip install plotly==4.4.1

!pip install chart_studio

#!pip install xlrd

#!pip install pandas_datareader

#!pip install seaborn

plt.style.use('Solarize_Light2')
import os

import pandas as pd

import numpy as np





def excel_to_df(excel_sheet):

 df = pd.read_excel(excel_sheet)

 df.dropna(how='all', inplace=True)



 index_PL = int(df.loc[df['Data provided by SimFin']=='Profit & Loss statement'].index[0])

 index_CF = int(df.loc[df['Data provided by SimFin']=='Cash Flow statement'].index[0])

 index_BS = int(df.loc[df['Data provided by SimFin']=='Balance Sheet'].index[0])



 df_PL = df.iloc[index_PL:index_BS-1, 1:]

 df_PL.dropna(how='all', inplace=True)

 df_PL.columns = df_PL.iloc[0]

 df_PL = df_PL[1:]

 df_PL.set_index("in million USD", inplace=True)

 (df_PL.fillna(0, inplace=True))

 



 df_BS = df.iloc[index_BS-1:index_CF-2, 1:]

 df_BS.dropna(how='all', inplace=True)

 df_BS.columns = df_BS.iloc[0]

 df_BS = df_BS[1:]

 df_BS.set_index("in million USD", inplace=True)

 df_BS.fillna(0, inplace=True)

 



 df_CF = df.iloc[index_CF-2:, 1:]

 df_CF.dropna(how='all', inplace=True)

 df_CF.columns = df_CF.iloc[0]

 df_CF = df_CF[1:]

 df_CF.set_index("in million USD", inplace=True)

 df_CF.fillna(0, inplace=True)

 

 df_CF = df_CF.T

 df_BS = df_BS.T

 df_PL = df_PL.T

    

 return df, df_PL, df_BS, df_CF



def combine_regexes(regexes):

 return "(" + ")|(".join(regexes) + ")"

#import tk_library_py

#from tk_library_py import excel_to_df
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
_, google_PL, google_BS, google_CF = excel_to_df("/kaggle/input/alphbet/SimFin-data .xlsx")

_, microsoft_PL, microsoft_BS, microsoft_CF = excel_to_df("/kaggle/input/microsoft/SimFin-data.xlsx")
google_BS
microsoft_BS
# Get the average annual stock price, which would be convenient to following calculation like target buy price and relevant several ratios

google_BS["Average Stock Price"]=[535.6230,568.9741,642.8156,884.2431,713.2142,619.9836,763.2142,939.7734,1122.0436,1191.2169]

# From the website: https://www.macrotrends.net/stocks/charts/MSFT/microsoft/stock-price-history

microsoft_BS["Average Stock Price"]=[27.0548,26.0522,29.8203,32.4915,42.4533,46.7136,55.2593,71.9840,101.0340,130.3820]

# From the website: https://www.macrotrends.net/stocks/charts/MSFT/microsoft/stock-price-history

google_BS["Average Stock Price"]
import chart_studio

chart_studio.tools.set_credentials_file(username='ShallweBuu', api_key='xJ6jcdZgMtiQu2y5vrGo')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes
# Book value can be defined as total asset minus total liabilities

google_BS["book value"] = google_BS["Total Assets"] - google_BS["Total Liabilities"]

microsoft_BS["book value"] = microsoft_BS["Total Assets"] - microsoft_BS["Total Liabilities"]

google_BS["book value"]
plt.bar(range(len(google_BS["Total Liabilities"])),google_BS["Total Liabilities"],label='Liabilities')

plt.bar(range(len(google_BS["Total Liabilities"])),google_BS["Total Equity"],bottom=google_BS["Total Liabilities"],label='Equity')

plt.plot(google_BS["Total Assets"],color='r')

plt.legend()
# I use Scatter to draw Assets line since "Assets = Equity + Liability"

# This can also reflect the book value of equity

assets = go.Scatter(

    x=google_BS.index,

    y=google_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=google_BS.index,

    y=google_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Bar(

    x=google_BS.index,

    y=google_BS["Total Equity"],

    name='Equity'

)



data = [assets, liabilities, shareholder_equity]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

fig_bs.update_layout(yaxis_title='USD', xaxis_title='Date', title='Assets, Liabilities and Equity')

fig_bs.show()

#py.iplot(fig_bs, filename='Total Assets and Liabilities',title='Assets, Liabilities and Equity')
fig, ax = plt.subplots()

plt.plot(google_BS["book value"],label="GOOGL")

plt.plot(microsoft_BS["book value"],label="MSFT")

plt.title('Book Value')

plt.xlabel('Date')

plt.ylabel('USD')

plt.legend()
# Calculate the book value per share

# Use the formula: book value per share = book value/shares outstanding

google_BS["shares_outstanding"]=[647,654,665,737,742,745,748,751,750,745] # from the website: https://www.macrotrends.net/stocks/charts/GOOGL/alphabet/shares-outstanding

microsoft_BS["shares_outstanding"]=[8927,8593,8506,8470,8399,8254,8013,7832,7794,7753] # from the website: https://www.macrotrends.net/stocks/charts/MSFT/microsoft/shares-outstanding

google_BS["Book value per share"]=google_BS["book value"]/google_BS["shares_outstanding"]

microsoft_BS["Book value per share"]=microsoft_BS["book value"]/microsoft_BS["shares_outstanding"]

google_BS["Book value per share"]
# Calculate the PB ratio

# Use the formula: PB ratio = Stock price/book value per share

google_BS["PB ratio"]=google_BS["Average Stock Price"]/google_BS["Book value per share"]

microsoft_BS["PB ratio"]=microsoft_BS["Average Stock Price"]/microsoft_BS["Book value per share"]

google_BS["PB ratio"]
# Look at the ROE of microsoft and see whether it has consistent high ROE 

# Use the formula: ROE = Net income/equity

google_BS["ROE"]=google_PL['Net Income Available to Common Shareholders']/google_BS['Total Equity']

microsoft_BS["ROE"]=microsoft_PL['Net Income Available to Common Shareholders']/microsoft_BS['Total Equity']

google_BS["ROE"]

# See the change of PB ratio

google_BS["PB ratio"].plot(label="GOOGL")

microsoft_BS["PB ratio"].plot(label="MSFT")

plt.title('PB Ratio')

plt.xlabel('Date')

plt.ylabel('Price-to-Book value Ratio')

plt.legend()
# See the change on ROE in last 10 years 

google_BS["ROE"].plot(label="GOOGL")

microsoft_BS["ROE"].plot(label="MSFT")

plt.title('ROE')

plt.xlabel('Date')

plt.ylabel('Return On Equity')

plt.legend()
# Choose the daily stock price in 2019 

import pandas as pd

import datetime

import pandas_datareader.data as web

from pandas import Series, DataFrame



start = datetime.datetime(2019, 1, 1)



end = datetime.datetime(2019, 12, 31)



google = web.DataReader("GOOGL", 'yahoo', start, end)



google.tail()
# I'm not sure this approach to get the margin of safety is correct

Average_Price=np.mean(google["Adj Close"])

Min_Price=np.min(google["Adj Close"])

safety=(Average_Price-Min_Price)/Average_Price

print("The margin of safety is", round(safety,4))
# I also search some information about the margin of safety

# Finally decide to use the 15% as the margin since Alphabet’s stocks are in the S&P 500 index

# From the website: https://trendshare.org/how-to-invest/what-is-a-margin-of-safety

margin_safety=0.15
google_PL
# Use iplot to present the grapg of revenue and see the tendency

grevenue = go.Scatter(

    x=google_PL.index,

    y=google_PL["Revenue"],

    name='GOOGL Rev'

)

mrevenue = go.Scatter(

    x=microsoft_PL.index,

    y=microsoft_PL["Revenue"],

    name='MSFT Rev'

)

data = [grevenue,mrevenue]

layout = go.Layout(

    barmode='stack'

)

fig_bs = go.Figure(data=data, layout=layout)

fig_bs.update_layout(yaxis_title='USD', xaxis_title='Date', title='Revenue')

fig_bs.show()

#py.iplot(fig_bs, filename='Revenue')
# Calculate the last 5 years Annual Revenue Growth

google_BS["Revenue Growth"]=google_PL["Revenue"]/google_PL["Revenue"].shift(1)-1

RevGrowth=np.mean(google_BS["Revenue Growth"].tail(5))

print("Annual Revenue Growth is",round(RevGrowth,4))
google_BS["Revenue Growth"]
if RevGrowth > 0.05:

    print("Annual Revenue Growth is larger than 0.05.")

else:

    print("Annual Revenue Growth is less than 0.05.")
# Display the three margin in the graph

# Use respective formula:

# gross margin = gross profit/revenue

# operating magin = operating income/revenue

# net margin = net income/revenue

google_PL["gross_margin"]=google_PL["Gross Profit"]/google_PL["Revenue"]

google_PL["operating_margin"]=google_PL["Operating Income (Loss)"]/google_PL["Revenue"]

google_PL["net_margin"]=google_PL["Net Income Available to Common Shareholders"]/google_PL["Revenue"]

microsoft_PL["gross_margin"]=microsoft_PL["Gross Profit"]/microsoft_PL["Revenue"]

microsoft_PL["operating_margin"]=microsoft_PL["Operating Income (Loss)"]/microsoft_PL["Revenue"]

microsoft_PL["net_margin"]=microsoft_PL["Net Income Available to Common Shareholders"]/microsoft_PL["Revenue"]

col_='''

gross_margin

operating_margin

net_margin

'''

margin_data=[]

for col in col_.strip().split("\n"):

    margin = go.Scatter(

        x=google_PL.index,

        y=google_PL[ col ],

        name="GOOGL_" + col

    )    

    margin_data.append(margin)





for col in col_.strip().split("\n"):

    margin = go.Scatter(

        x=microsoft_PL.index,

        y=microsoft_PL[ col ],

        name="MSFT_" + col

    )    

    margin_data.append(margin)

        

layout_margin = go.Layout(

    barmode='stack'

)



fig_margin = go.Figure(data=margin_data, layout=layout_margin)

fig_margin.update_layout(yaxis_title='Margin rate', xaxis_title='Date', title='Margin Pressure')

fig_margin.show()                     

#py.iplot(fig_margin, filename='margin')
# EPS = Net Income Available to Common Shareholders/Total weighted average shares outstanding



google_BS["EPS"]=google_PL["Net Income Available to Common Shareholders"]/google_BS["shares_outstanding"]

microsoft_BS["EPS"]=microsoft_PL["Net Income Available to Common Shareholders"]/microsoft_BS["shares_outstanding"]
# Display the historical EPS

EPS1 = go.Scatter(

    x=google_BS.index,

    y=google_BS["EPS"],

    name='GOOGL EPS'

)

EPS2 = go.Scatter(

    x=microsoft_BS.index,

    y=microsoft_BS["EPS"],

    name='MSFT EPS'

)

data = [EPS1,EPS2]

layout = go.Layout(

    barmode='stack'

)

fig_bs = go.Figure(data=data, layout=layout)

fig_bs.update_layout(yaxis_title='USD', xaxis_title='Date', title='EPS')

fig_bs.show()

#py.iplot(fig_bs, filename='EPS')
# See the growth of EPS

google_BS["EPS Growth"]=google_BS["EPS"]/google_BS["EPS"].shift(1)-1

google_BS["EPS Growth"]
# Calculate the interest coverage

# Formula: Interest Coverage = EBIT/Interest Expense

EBIT=[19360000,23716000,26178000,27524000,34231000] # From: https://www.macrotrends.net/stocks/charts/GOOG/alphabet/ebit

IE=[104000,124000,109000,114000,100000] 

InterestCoverage=[a/b for a,b in zip(EBIT,IE)]

# Data from the website: 

# Operating Income(TTM) can be regarded as EBIT and select the data of Interest Expense(TTM)

# Also the data from last 5 years

InterestCoverage_=[round(i,4) for i in InterestCoverage]

_InterestCoverage=min(InterestCoverage_)

print("The Interest Coverage is",_InterestCoverage) # I Choose the minimum value in past five years
# List the annual interest coverage in recent 5 years

InterestCoverage_
# There are 5 steps to get the target buy price

#1. Earnings Per Share (EPS) Annual Compounded Growth Rate

#2. Estimate EPS 10 years from now

#3. Determine Current Target Buy Price

#4. Margin of Safety (25% off the target buy price)

#5. Debt to Equity Ratio
# To calculate the EPS Annual Compounded Growth Rate

# Firstly get the EPS from last 10 years



google_BS["EPS"]
# Using the formula: Growth rate=(future value/present value)^(1/years)-1

def CAGR(PV,FV,Y):

    r=(FV/PV)**(1/Y)-1

    return r



PV=google_BS["EPS"].loc["FY '10"]

FV=google_BS["EPS"].loc["FY '19"]

Y=9

print("The CAGR of microsoft is",round(CAGR(PV,FV,Y),4))
# Using the formula: Future value = present value * (1+growth rate)^years

def EPS10(PV_,GR,Y_):

    FV_=PV_*(1+GR)**Y_

    return FV_

PV_=google_BS["EPS"].loc["FY '19"]

GR=CAGR(PV,FV,Y)

Y_=10

print("The eps 10 years from now is", round(EPS10(PV_,GR,Y_),4))
# To determine the future stock price in 10 years we need to calculate the Average PE Ratio at first

# Using the formula: PE ratio=Stock price/EPS

google_PL["Average PE Ratio"]=google_BS["Average Stock Price"]/google_BS["EPS"]

Average=np.mean(google_PL["Average PE Ratio"])

print("The Average PE Ratio of last 10 years is",round(Average,4))
# Get the Stock price 10 years from now. Formula: Stock Price = Estimated Future EPS * Average PE Ratio

StockPrice=EPS10(PV_,GR,Y_)*Average

print("Stock Price 10 Years from now is",round(StockPrice,4))
# Get the buy price: buy price= future value/(1+discount rate)^years

def BuyPrice(fv,year,dr):

    price=fv/(1+dr)**year

    return price

fv=StockPrice

year=10

dr=0.0693

print("The target buy price is",round(BuyPrice(fv,year,dr),4))
# Add 15% of margin of safety 

# https://trendshare.org/how-to-invest/what-is-a-margin-of-safety

FinalPrice=BuyPrice(fv,year,dr)*(1-0.15)

print("add at 15% margin of safety:",round(FinalPrice,4))
# 1.Present EPS under the target buy price

# 2.Compared to the industry PE ratio

# 3.Get the sell price
# First confirm the present EPS with target buy price

present_EPS=FinalPrice/Average

present_EPS
# The tech software industry PE is 40.25 from the website: https://csimarket.com/Industry/Industry_Valuation.php?s=1000

# Sell price = Present EPS * Industry PE ratio

selling_price=present_EPS*40.25

print("Once the price rise to",round(selling_price,2),", I would sell the stock.")
# See whether ROE is consistent and more than 15%

a=False

l=[]

for i in google_BS["ROE"]:    

    if i<0.15:

        a=True

if a==False:

    print('google has consistent high ROE and more than 15%.')

else:

    for i in google_BS["ROE"]:

        if i<0.15:

            l.append(i)

    x=len(l)

    if x==1:

        print('google did not reach the required value',x,'time in last 10 years.')

    if x>1:

        print('google did not reach the required value',x,'times in last 10 years.')

# Look at the ROA of microsoft and see if it has consistent high ROA

google_BS["ROA"]=google_PL['Net Income Available to Common Shareholders']/google_BS['Total Assets']

google_BS["ROA"]
# From above data, microsoft has had ROA lower than 0.07 in 2015 and 2018

a=False

l=[]

for i in google_BS["ROA"]:    

    if i<0.07:

        a=True

if a==False:

    print('google has consistent high ROA.')

else:

    for i in google_BS["ROA"]:

        if i<0.07:

            l.append(i)

    x=len(l)

    if x==1:

        print('google did not reach the required value',x,'time in last 10 years.')

    if x>1:

        print('google did not reach the required value',x,'times in last 10 years.')
# But the average ROA in last 10 years is larger than 0.07

sum=0

for i in google_BS["ROA"]:

    sum=sum+i

average=sum/len(google_BS["ROA"])

print("the average ROA of google in past 10 years is",round(average,4))
# Compare ROA and ROE

ROA = go.Scatter(

    x=google_BS.index,

    y=google_BS["ROA"],

    name='ROA'

)

ROE = go.Scatter(

    x=google_BS.index,

    y=google_BS["ROE"],

    name='ROE'

)

data = [ROA,ROE]

layout = go.Layout(

    barmode='stack'

)

fig_bs = go.Figure(data=data, layout=layout)

fig_bs.update_layout(yaxis_title='Return', xaxis_title='Date', title='ROA & ROE')

fig_bs.show()

#py.iplot(fig_bs, filename='ROA&ROE')
# Compare the 5 times net income and the long term debt.

google_BS["comparison net income"]=5*google_PL["Net Income Available to Common Shareholders"]-google_BS["Long Term Debt"]

a=False

for i in google_BS["comparison net income"]:

    if i<0:

        a=True

if a==False:

    print("5 times Net Income > Long Term Debt")

else:

    print("5 times Net Income < Long Term Debt")
google_BS["comparison net income"]
# Calculate the DE Ratio

google_BS["DE Ratio"]=google_BS["Total Liabilities"]/google_BS["Total Equity"]

google_BS["DE Ratio"]

# Plot GOOGL close price in last 10 years

start = datetime.datetime(2010, 1, 1)



end = datetime.datetime(2020, 12, 31)



google = web.DataReader("GOOGL", 'yahoo', start, end)



google.tail()



google['Close'].plot()

plt.title('GOOGL Close Price')



# Resampling for monthly data

google_month = google['Close'].resample('M').mean()



train_data = google_month['2010':'2019']



train_data.plot()

plt.title('Monthly Close Price')

plt.show()

# Take the first difference of the data

from statsmodels.tsa.stattools import adfuller

diff_data = train_data.diff().dropna()

result = adfuller(diff_data)



plt.figure()

plt.plot(diff_data)

plt.title('First Difference')

plt.xlabel('Date')

plt.show()



print('ADF Statistic:', result[0])

print('p-value:', result[1])

print(result)
!pip install pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Import pmdarima as pm

import pmdarima as pm

import warnings

warnings.filterwarnings('ignore')
# Create auto_arima model

model = pm.auto_arima(train_data,

                      seasonal=True, m=12,

                      d=1, D=1, 

                 	  max_p=2, max_q=2,

                      trace=True,

                      error_action='ignore',

                      suppress_warnings=True)

                       

# Print model summary

print(model.summary())
# Input the SARIMAX model

model = SARIMAX(train_data, order = (2,1,2), seasonal_order = (0, 1, 1, 12))

# Fit model

results = model.fit()

# Generate predictions for recent 1 year

forecast = results.get_prediction(start='2018-12-31')

mean_forecast=results.predict(start='2018-12-31')



# Get confidence intervals of predictions

confidence_intervals = forecast.conf_int()



# # Print best estimate predictions

display(mean_forecast, confidence_intervals)
# plot the Google data

plt.plot(train_data.index, train_data, label='observed')



# plot mean predictions

mean_forecast.plot(color='r',figsize=(12,6))



# shade the area between confidence limits

plt.fill_between(confidence_intervals.index, 

                 confidence_intervals.iloc[:, 0],

                 confidence_intervals.iloc[:, 1], color='grey')



# set labels, legends and show plot

plt.title('Forecast & Observed')

plt.xlabel('Date')

plt.ylabel('Google Stock Price - Close USD')

plt.legend()

plt.show()
# Create the 4 diagostics plots

results.plot_diagnostics(figsize=(12, 12))

plt.show()
# Get forecast one year ahead in future

pred_uc = results.get_forecast(steps=12)

 

# Get confidence intervals of forecasts

pred_ci = pred_uc.conf_int()



print(pred_uc,pred_ci)
# Plot the predicted stock price

ax = train_data.plot(label='Observed', figsize=(12, 6))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast',color='r')

# Fill the missing data accordingly

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Google Stock Price - Close USD')

 

plt.legend(loc=4)

plt.show()
# Compare with MSFT, AAPL, AMZN, GE， IBM

dfcom = web.DataReader(['GOOGL','MSFT','AAPL','GE','IBM'],'yahoo',start=start,end=end)['Close']

print(dfcom.tail())
# Use percent change to define the returns

retscom=dfcom.pct_change()

# See the correalation between these stocks

corr=retscom.corr()

print(corr)
# Compare the GOOGL and MSFT

plt.scatter(retscom.GOOGL,retscom.MSFT)

plt.xlabel('Returns GOOGL')

plt.ylabel('Returns MSFT')
# Use KDE to generate estimation of the distributions

# Refer to: https://en.wikipedia.org/wiki/Kernel_density_estimation

from pandas import Series, DataFrame

pd.plotting.scatter_matrix(retscom,diagonal='kde',figsize=(10,10))
# Use hot maps to validate the correlation

plt.imshow(corr, cmap='hot', interpolation='none')

plt.colorbar()



plt.xticks(range(len(corr)),corr.columns)

plt.yticks(range(len(corr)),corr.columns)
# Average return as the expected returns

# Standard Deviation as the risks

x1=-0.001

x2=0.001

y1=0.01

y2=0.02

plt.xlim(x1, x2)

plt.ylim(y1, y2)

plt.scatter(retscom.mean(),retscom.std())





plt.xlabel('Expected returns')

plt.ylabel('Risk')

for label, x, y in zip(retscom.columns, retscom.mean(), retscom.std()):

    plt.annotate(

        label, 

        xy = (x, y), xytext = (20, -20),

        textcoords = 'offset points', ha = 'right', va = 'bottom',

        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),

        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    

plt.show()