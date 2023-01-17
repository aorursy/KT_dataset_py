#Bakyt Nursaiyn 19453647
#Assignment 2
#Value Investing Stock Analysis with Python
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

import os
# pip list
#comment out all the pip install
#!pip install plotly==4.4.1
#!pip install chart_studio
#!pip install xlrd
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))# Read IBM_Fin_Stat.xlsx into a pandas dataframe# Read IBM_Fin_Stat.xlsx into a pandas dataframe
_, intc_PL, intc_BS, intc_CF = excel_to_df("/kaggle/input/Intel_simfin.xlsx")
intc_BS
del(intc_BS['Assets'])
intc_BS
intc_BS["_Total Current Assets"] = intc_BS["Cash, Cash Equivalents & Short Term Investments"] + intc_BS["Accounts & Notes Receivable"] + intc_BS["Inventories"] + intc_BS["Other Short Term Assets"]
intc_BS[["_Total Current Assets", "Total Current Assets"]]
intc_BS["_NonCurrent Assets"] = intc_BS["Property, Plant & Equipment, Net"] + intc_BS["Long Term Investments & Receivables"] + intc_BS["Other Long Term Assets"]
intc_BS["_Total Assets"] = intc_BS["_NonCurrent Assets"] + intc_BS["_Total Current Assets"] 
intc_BS["_Total Liabilities"] = intc_BS["Total Current Liabilities"] + intc_BS["Total Noncurrent Liabilities"]
intc_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline
intc_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
intc_BS[ asset_columns ].plot()
!pip install chart_studio
import chart_studio
chart_studio.tools.set_credentials_file(username='nursaiynb', api_key='G3BdeHhGUNcPHafQdX2Q')
import chart_studio.plotly as py
import plotly.graph_objs as go
assets = go.Bar(
    x=intc_BS.index,
    y=intc_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=intc_BS.index,
    y=intc_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=intc_BS.index,
    y=intc_BS["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
fig_bs.show()
asset_data = []
columns = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
Property, Plant & Equipment, Net
Long Term Investments & Receivables
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=intc_BS.index,
        y=intc_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
fig_bs_assets.show()
liability_data = []
columns = '''
Payables & Accruals
Short Term Debt
Other Short Term Liabilities
Long Term Debt
Other Long Term Liabilities
'''


for col in columns.strip().split("\n"):
    liability_bar = go.Bar(
        x=intc_BS.index,
        y=intc_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
fig_bs_liabilitys.show()
intc_BS["Working Capital"] = intc_BS["Total Current Assets"] - intc_BS["Total Current Liabilities"]
intc_BS[["Working Capital"]].plot()
intc_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 

PR_data = []
columns = '''
Accounts & Notes Receivable
Payables & Accruals
'''

for col in columns.strip().split("\n"):
    PR_Scatter = go.Scatter(
        x=intc_BS.index,
        y=intc_BS[ col ],
        name=col
    )    
    PR_data.append(PR_Scatter)
    
layout_PR = go.Layout(
    barmode='stack'
)

fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)
fig_bs_PR.show()
intc_BS["Inventories"].plot()
intc_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
# Using Plotly

AAA_data = []
columns = '''
Property, Plant & Equipment, Net
Long Term Investments & Receivables
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    AAA_bar = go.Bar(
        x=intc_BS.index,
        y=intc_BS[ col ],
        name=col
    )    
    AAA_data.append(AAA_bar)
    
layout_AAA = go.Layout(
    barmode='stack'
)

fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)
fig_bs_AAA.show()
equity_columns = '''
Preferred Equity
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
'''

equity_columns = [ x for x in equity_columns.strip().split("\n")]
equity_columns
intc_BS[equity_columns].plot()
# Using Plotly

equity_data = []
columns = '''
Preferred Equity
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
'''


for col in columns.strip().split("\n"):
    equity_Scatter = go.Scatter(
        x=intc_BS.index,
        y=intc_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)
fig_bs_equity.show()
# According to simfin data, Intel has a preferred stock, but no intengible assets, and no goodwill, so it is deducted from Assets and Liabilities

intc_BS["book value"] = intc_BS["Total Assets"] - intc_BS["Total Liabilities"] - intc_BS["Preferred Equity"]
intc_BS["book value"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  
#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate
#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp

PE_RATIO = 9.36 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/85652

# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/intc/earnings-growth
GROWTH_RATE = 0.07209 # Forcast over the next five years

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("Intel Corp's PEG Ratio is", PEG_ratio)
#Additional calculation part is given below. All the information sources and further comments are given in respective code blocks. 
#The next step is to calculate annual compounded growth rate of EPS
#Earnings and average shares outstanding figures were obtained from simfin.com
earningsTTM10, earningsTTM = 10994000000, 15973000000
avsharesTTM10, avsharesTTM = 5645000000, 4473000000 
epsTTM10 = earningsTTM10 / avsharesTTM10
epsTTM = earningsTTM / avsharesTTM
CAGR = (epsTTM/epsTTM10)**(1/9) - 1
print(CAGR)
#Now we need to find EPS 10 years from now. For that, we will use CAGR formula and extract epsTTM from it, making it the EPS value 10 years from now.
eps10 = (1+CAGR)**10*epsTTM
print(eps10)
#To calculate the future price 10 years from now, we multiply future eps by average PE ratio in between 2006 and 2019, obtained from macrotrends.net
aver_pe = 11.76
future_price = eps10 * aver_pe
print(future_price)
#To calculate the target buy price, we discount the future price back to now using 6.5% average WACC for computer services companies obtained from http://people.stern.nyu.edu/adamodar/New_Home_Page/datafile/wacc.htm
target_buy = future_price / (1 + 0.065) ** 10
print(target_buy)
mar_safety = target_buy * 0.75 #per assignment instructions, the margin of safety is 25% off the target buy price.
print(mar_safety)
debt_to_equity = intc_BS['Total Liabilities'] / intc_BS['Total Equity']
print(debt_to_equity)
int_exp = 488 #obtained from annual report, the interest expense is $723 million
int_cov = intc_PL['Pretax Income (Loss)'] / int_exp #please consider only FY '19 figure
print(int_cov)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime
# The tech stocks we'll use for this analysis
tech_list = ['IBM', 'INTC']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)


#For loop for grabing yahoo finance data and setting as a dataframe
for stock in tech_list:   
    # Set DataFrame as the Stock Ticker
    globals()[stock] = DataReader(stock, 'yahoo', start, end)
# for company, company_name in zip(company_list, tech_list):
#     company["company_name"] = company_name
company_list = [IBM, INTC]
company_name = ["IBM", "Intel"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
df.sample(10)
IBM.describe()
INTC.describe()
IBM.info()
INTC.info()
# Let's see a historical view of the closing price


plt.figure(figsize=(12, 8))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{tech_list[i - 1]}")
# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(12, 8))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"{tech_list[i - 1]}")
ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

IBM[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0])
axes[0].set_title('IBM')

INTC[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1])
axes[1].set_title('INTEL')

fig.tight_layout()
# We'll use pct_change to find the percent change for each day
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

IBM['Daily Return'].plot(ax=axes[0], legend=True, linestyle='--', marker='o')
axes[0].set_title('IBM')

INTC['Daily Return'].plot(ax=axes[1], legend=True, linestyle='--', marker='o')
axes[1].set_title('Intel')

fig.tight_layout()
# Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
plt.figure(figsize=(12, 12))

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    sns.distplot(company['Daily Return'].dropna(), bins=100, color='purple')
    plt.ylabel('Daily Return')
    plt.title(f'{company_name[i - 1]}')
# Could have also done:
#IBM['Daily Return'].hist()
# Grab all the closing prices for the tech stock list into one DataFrame
closing_df = DataReader(tech_list, 'yahoo', start, end)['Adj Close']

# Let's take a quick look
closing_df.head()
# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
tech_rets.head()
# We'll use joinplot to compare the daily returns of Google and Microsoft
sns.jointplot('IBM', 'INTC', tech_rets, kind='scatter')
# We can simply call pairplot on our DataFrame for an automatic visual analysis 
# of all the comparisons

sns.pairplot(tech_rets, kind='reg')
# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(tech_rets.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) 
# or the color map (BluePurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
return_fig.map_diag(plt.hist, bins=30)

# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
returns_fig = sns.PairGrid(closing_df)

# Using map_upper we can specify what the upper triangle will look like.
returns_fig.map_upper(plt.scatter,color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
returns_fig.map_diag(plt.hist,bins=30)
# Let's go ahead and use seaborn for a quick correlation plot for the daily returns
sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
# Let's start by defining a new DataFrame as a clenaed version of the oriignal tech_rets DataFrame
rets = tech_rets.dropna()

area = np.pi*20

plt.figure(figsize=(12, 10))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
#Get the stock quote
df = DataReader('INTC', data_source='yahoo', start='2012-01-01', end='2020-02-23')
#Show the data
df
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
#Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .8 ))

training_data_len
#Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data
#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
# Convert the data to a numpy array
x_test = np.array(x_test)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
#Show the valid and predicted prices
valid