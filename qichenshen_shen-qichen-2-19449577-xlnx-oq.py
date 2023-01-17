#!pip install xlrd

!pip install simfin

!pip install plotly==4.4.1

!pip install chart_studio

import pandas_datareader.data as web

import datetime

import os

import pandas as pd

import numpy as np

import chart_studio

import chart_studio.plotly as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import plot
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
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
currentfile = r'/kaggle/input/19449577-assignment2/SimFin-data.xlsx'

_, XLNX_PL, XLNX_BS, XLNX_CF = excel_to_df(currentfile)
del(XLNX_BS["Assets"])

XLNX_BS
XLNX_PL
xlnx = pd.read_csv(r'/kaggle/input/19449577-assignment2/whole_income_annual.csv'

                   ,sep=';',index_col = 0)

# because of the lack of the 2019 data, I find them on Bloomberg manually

xlnx.loc['XLNX'].to_excel('XLNX_statement.xlsx')
xlnx_fs = pd.read_excel(r'/kaggle/input/19449577-assignment2/XLNX_statement2.xlsx')

xlnx_fs
EBT = xlnx_fs['Net Income'] - xlnx_fs['Income Tax (Expense) Benefit, Net']

EBIT = EBT + xlnx_fs['Interest Expense, Net']

EBITDA = EBIT + xlnx_fs['Depreciation & Amortization']
#add price in the financial statement

xlnx_fs['Price'] = 25.68,32.15,36.48,38.17,53.13,42.06,47.62,57.89,72.24,126.79

market_value = xlnx_fs['Price'] * xlnx_fs['Shares (Diluted)'] / 1000000

market_value
XLNX_BS['Market Value'] = 7112,8618,9928,10404,15269,11613,12793,15561,18635,32322

XLNX_BS['EBITDA'] = 426,796,620,567,726,750,646,689,735,939

XLNX_BS['EBIT'] = 429,797,628,577,735,760,652,694,738,944
EV = XLNX_BS['Market Value'] + XLNX_BS['Total Current Liabilities'] + XLNX_BS['Share Capital & Additional Paid-In Capital'] - XLNX_BS["Cash, Cash Equivalents & Short Term Investments"]

EV
EVEBITDA = EV / XLNX_BS['EBITDA']

print(EVEBITDA)

print('*********')

print(EVEBITDA.mean())
peer_value = pd.read_excel(r'/kaggle/input/19449577-assignment2/Ratio_Historical.xlsx')

peer_value
peer_corr = peer_value.corr()

peer_corr
plt.imshow(peer_corr, cmap='hot', interpolation='none')

plt.colorbar()

plt.xticks(range(len(peer_corr)), peer_corr.columns,rotation = 45)

plt.yticks(range(len(peer_corr)), peer_corr.columns)
internal = {'Cash': XLNX_BS['Cash, Cash Equivalents & Short Term Investments'],

           'Other Long Term Assets': XLNX_BS['Other Long Term Assets'],

           'Accounts & Notes Receivable': XLNX_BS['Accounts & Notes Receivable'],

           'Long Term Debt': XLNX_BS['Long Term Debt'],

           'Retained Earnings': XLNX_BS['Retained Earnings']}

internal_frame = pd.DataFrame(internal)

internal_newframe = internal_frame.iloc[5:,:]

internal_newframe
internal_newframe_corr = internal_newframe.corr()

sns.heatmap(internal_newframe_corr)
#calculate FCFF

FCFF = XLNX_BS['EBIT']  + XLNX_BS['EBIT'] * (1 - 0.152) + XLNX_CF['Depreciation & Amortization'] - XLNX_CF['Change in Fixed Assets & Intangibles'] - XLNX_CF['Change in Working Capital']

FCFF
#calculate FCFF growth, the first step is to get the effective tax rate 

EITR = abs(xlnx_fs['Income Tax (Expense) Benefit, Net']) / EBT

EITR = EITR.mean()

EITR
#get ROIC

NOPAT = XLNX_BS['EBIT'] * 0.84

InvestedCapital = XLNX_BS['Short Term Debt'] + XLNX_BS['Other Short Term Liabilities'] + XLNX_BS['Total Equity']

ROIC = NOPAT / InvestedCapital

ROIC.mean()
#get retention rate 

RR = 1 - (abs(XLNX_CF['Dividends Paid']) + abs(XLNX_CF['Cash From (Repayment of) Debt'])) / XLNX_BS['EBIT']*(1 - EITR)

RR
#get FCFF growth

FCFF_growth = RR * ROIC.mean()

FCFF_growth
DE = (XLNX_BS["Total Liabilities"] - XLNX_BS["Total Noncurrent Liabilities"]) / XLNX_BS["Total Equity"]

DE
#calculate the Debt and Equity percentage

D_percent = 1/(1/DE + 1)

E_percent = 1/(DE + 1)
#calculate WACC to discount, we use the short-term debt rate 3.05% and rate of equity 9.64%

WACC = 0.035 * D_percent + 0.0964 * E_percent

WACC
print(D_percent.mean())

print(E_percent.mean())

print(WACC.mean())
FCFF_est = []

for i in range(0,10):

    fcff = FCFF[-1] * pow(1+FCFF_growth[-1],i)

    FCFF_est.append(fcff)

print(FCFF_est)
discount_factor = []

for i in range(0,10):

    discount_rate = pow(1+WACC.mean(),i)

    discount_factor.append(discount_rate)

print(discount_factor)
#set dataframe

DF = {'FCFF_EST':FCFF_est,'DR':discount_factor}

Framework = pd.DataFrame(DF)

Framework
Framework['PV'] = Framework['FCFF_EST'] / Framework['DR']

EV2 = Framework['PV'].sum()

EV2
#calcualte the average EV, EV2 calculated by EBIT discount, EV calculate by EV/EBITDA

EV_Avg = (EV+EV2) / 2

EV_Avg
#calculate PS Ratio

xlnx_ps = xlnx_fs['Price'] / xlnx_fs['Revenue']/1000000

xlnx_ps.astype('float64')
#calculate Gross Profit Margin

XLNX_PL['Gross_profit_margin'] = (XLNX_PL['Revenue'] / XLNX_PL['Gross Profit']) -1

XLNX_PL['Gross_profit_margin'] 
#calculate EVA

EVA = NOPAT - WACC * InvestedCapital

EVA
compare_ratio = pd.read_excel(r'/kaggle/input/19449577-assignment2/XLNX.xlsx')

compare_TXN = pd.read_excel(r'/kaggle/input/19449577-assignment2/Compare.xlsx',index_col=0)

compare_ratio.head()
compare_ratio.columns=['Ticker','Name','Sales Growth','EBITDA Growth','EBITDA Margin','Operating Income Margin',

                     'Net Income Growth','Net Profit Margin','Capex/Sales','Return on Invested Capital','Return on Assets','Return on Equity']

compare_ratio = compare_ratio.dropna()

compare_ratio = compare_ratio.drop(index=1)

compare_ratio = compare_ratio.drop(['Ticker'], axis=1)

compare_ratio = compare_ratio.drop(['EBITDA Margin', 'Net Profit Margin','Capex/Sales','Net Income Growth'], axis=1)

compare_ratio = compare_ratio.drop(compare_ratio.index[[2,3,5,6,7,8]])

compare_ratio
compare_TXN
DE_frame = compare_TXN[['TXN D/E','XLNX D/E','Average D/E']]

ROIC_frame = compare_TXN[['TXN ROIC','XLNX ROIC','Average ROIC']]

PS_frame = compare_TXN[['TXN P/S','XLNX P/S','Average P/S']]

EV_frame = compare_TXN[['TXN EV/EBITDA','XLNX EV/EBITDA','Average EV/EBITDA']]

DE_frame
DE_frame['XLNX D/E']
from plotly.subplots import make_subplots

fig = make_subplots(

    rows=2, cols=2,

    subplot_titles=("DE Ratio", "ROIC Ratio", "P/S Ratio", "EV/EBITDA Ratio"))



fig.add_trace(go.Scatter(x=DE_frame.index, y=DE_frame['TXN D/E'],name='TXN D/E')

              ,row=1, col=1)

fig.add_trace(go.Scatter(x=DE_frame.index, y=DE_frame['XLNX D/E'],name='XLNX D/E'),

              row=1, col=1)

fig.add_trace(go.Scatter(x=DE_frame.index, y=DE_frame['Average D/E'],name='Average D/E'),

              row=1, col=1)

fig.add_trace(go.Scatter(x=ROIC_frame.index, y=ROIC_frame['TXN ROIC'],name='TXN ROIC'),

              row=1, col=2)

fig.add_trace(go.Scatter(x=ROIC_frame.index, y=ROIC_frame['XLNX ROIC'],name='XLNX ROIC'),

              row=1, col=2)

fig.add_trace(go.Scatter(x=ROIC_frame.index, y=ROIC_frame['Average ROIC'],name='Average ROIC'),

              row=1, col=2)



fig.add_trace(go.Scatter(x=PS_frame.index, y=PS_frame['TXN P/S'],name='TXN P/S'),

              row=2, col=1)

fig.add_trace(go.Scatter(x=PS_frame.index, y=PS_frame['XLNX P/S'],name='XLNX P/S'),

              row=2, col=1)

fig.add_trace(go.Scatter(x=PS_frame.index, y=PS_frame['Average P/S'],name='Average P/S'),

              row=2, col=1)



fig.add_trace(go.Scatter(x=EV_frame.index, y=EV_frame['TXN EV/EBITDA'],name='TXN EV/EBITDA'),

              row=2, col=2)

fig.add_trace(go.Scatter(x=EV_frame.index, y=EV_frame['XLNX EV/EBITDA'],name='XLNX EV/EBITDA'),

              row=2, col=2)

fig.add_trace(go.Scatter(x=EV_frame.index, y=EV_frame['Average EV/EBITDA'],name='Average EV/EBITDA'),

              row=2, col=2)



fig.update_layout(height=600, width=1100,

                  title_text="Multiple Ratio")



fig.show()
py.sign_in('thxq24302', 'XM84hNdT1KJxmwlkTthx')

EBIT = go.Bar(

    x=XLNX_BS.index,

    y=XLNX_BS["EBIT"],

    name='EBIT'

)

EBITDA = go.Bar(

    x=XLNX_BS.index,

    y=XLNX_BS["EBITDA"],

    name='EBITDA'

)

Gross_profit = go.Bar(

    x=XLNX_BS.index,

    y=XLNX_PL['Gross Profit'],

    name='Gross_profit'

)

EVA_value = go.Bar(

    x=XLNX_BS.index,

    y=EVA.values,

    name='EVA'

)



data = [EBIT, EBITDA, Gross_profit, EVA_value]

layout = go.Layout(barmode='group')



Profitability = go.Figure(data=data, layout=layout)

Profitability.show()
#XLNX self ratio

DE_Ratio = go.Scatter(

    x=XLNX_BS.index,

    y=DE,

    name='DE'

)

ROIC_value = go.Scatter(

    x=XLNX_BS.index,

    y=ROIC,

    name='ROIC'

)

Gross_profit_margin = go.Scatter(

    x=XLNX_BS.index,

    y=XLNX_PL['Gross_profit_margin'],

    name='Gross_profit_margin'

)



data = [DE_Ratio, ROIC_value, Gross_profit_margin]

layout = go.Layout(barmode='group')



fig_plot = go.Figure(data=data, layout=layout)

fig_plot.show()
from pandas_datareader import data

start_date = '2009-01-01'

end_date = '2019-12-23'

xlnxdata = data.DataReader('TXN', 'yahoo', start_date, end_date)

NASDAQdata = data.DataReader('^IXIC', 'yahoo', start_date, end_date)

print(xlnxdata.info())

print('***********')

print(xlnxdata.info())
trace1 = go.Scatter(

    x=NASDAQdata.index,

    y=xlnxdata['Adj Close'],

    name='XLNX'

)

trace2 = go.Scatter(

    x=NASDAQdata.index,

    y=NASDAQdata['Adj Close'],

    name='NASDAQ',

    xaxis='x', 

    yaxis='y2'

)

 

data = [trace1, trace2]

layout = go.Layout(

    yaxis2=dict(anchor='x', overlaying='y', side='right')

)

 

tracevalue = go.Figure(data=data, layout=layout)

tracevalue.show()
xlnxdata.shape
xlnx_frame = xlnxdata.loc[:,['Adj Close']]

xlnx_frame['high_low_Per'] = (NASDAQdata['High'] - NASDAQdata['Low']) / NASDAQdata['Close'] * 100.0

xlnx_frame['NASDAQ_change'] = (NASDAQdata['Adj Close'] - NASDAQdata['Open']) / NASDAQdata['Open'] * 100.0

xlnx_frame['NASDAQ_Volume'] = NASDAQdata['Volume']

xlnx_frame.tail()
xlnx_corr = xlnx_frame.corr()

print(xlnx_corr)
sns.pairplot(xlnx_frame, x_vars=['high_low_Per','NASDAQ_change','NASDAQ_Volume'], y_vars='Adj Close', height=6, aspect=0.7)
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import statsmodels.api as sm
plt.figure(figsize=(12,6))

plt.tight_layout()

sns.distplot(xlnx_frame['Adj Close'])
#inspect null value

xlnx_frame.isnull().any()
value_save = {'xlnxAdj_Close': xlnxdata['Adj Close'],

           'NASDAQAdj_Close': NASDAQdata['Adj Close']}

xlnx_data = pd.DataFrame(value_save)

xlnx_data.head()
xlnx_data['xlnxAdj_Close'].describe()
xlnx_data['NASDAQAdj_Close'].describe()
#calculate the daily return

daily_return = (xlnx_data.diff()/xlnx_data.shift(periods = 1)).dropna()

daily_return.head()
#observe the abnormal value

daily_return[daily_return['xlnxAdj_Close'] > 0.105]
#observe the return

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

daily_return['xlnxAdj_Close'].plot(ax=ax[0])

ax[0].set_title('xlnxAdj_Close')

daily_return['NASDAQAdj_Close'].plot(ax=ax[1])

ax[1].set_title('NASDAQAdj_Close')
#Whether the yield follows normal distribution

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

sns.distplot(daily_return['xlnxAdj_Close'],ax=ax[0])

ax[0].set_title('xlnxAdj_Close')

sns.distplot(daily_return['NASDAQAdj_Close'],ax=ax[1])

ax[1].set_title('NASDAQAdj_Close')
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))

plt.scatter(daily_return['xlnxAdj_Close'],daily_return['NASDAQAdj_Close'])

plt.title('Scatter Plot of daily return between XLNX and NASDAQ')
daily_return['intercept']=0.5
model = sm.OLS(daily_return['xlnxAdj_Close'],daily_return[['NASDAQAdj_Close','intercept']])

results = model.fit()

results.summary()
from pandas_datareader import data

from statsmodels.tsa.arima_model import ARIMA
start_date = '2015-01-01'

end_date = '2020-02-01'

stockdata = data.DataReader('XLNX', 'yahoo', start_date, end_date)

print(xlnxdata.info())

print('***********')

xlnxdata.head()
xlnx_week = stockdata['Adj Close'].resample('W-WED').mean()

xlnx_train = xlnx_week['2015':'2020']

xlnx_train.head()
xlnx_train.plot(figsize=(10,6))

plt.legend(loc='best')

plt.title('Adj Close')

sns.despine()
xlnx_diff = xlnx_train.diff(2)



plt.plot(xlnx_diff)

plt.title('second_diff')

plt.show()
xlnx_diff.dropna(inplace=True)

fig = plt.figure(figsize=(12,8))

ax1=fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(xlnx_diff,lags=30,ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(xlnx_diff,lags=30,ax=ax2)
arma_mod50 = sm.tsa.ARMA(xlnx_diff,(5,0)).fit()

print("arma_mod50:",arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

arma_mod01 = sm.tsa.ARMA(xlnx_diff,(0,1)).fit()

print("arma_mod01:",arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)

arma_mod51 = sm.tsa.ARMA(xlnx_diff,(5,1)).fit()

print("arma_mod51:",arma_mod51.aic,arma_mod51.bic,arma_mod51.hqic)
resid = arma_mod51.resid
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=30, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(resid, lags=30, ax=ax2)
print(sm.stats.durbin_watson(resid))
from statsmodels.graphics.api import qqplot

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(111)

fig = qqplot(resid, line='q', ax=ax, fit=True)
model = ARIMA(xlnx_train, order=(0,2,1),freq='W-WED')
result = model.fit()

print(result.summary())
pred = result.predict('2020','2021',dynamic=True,typ='levels')

print(pred.tail())
plt.figure(figsize=(10,6))

plt.xticks(rotation=45)

plt.plot(pred)

plt.plot(xlnx_train)

plt.legend(['pred','xlnx_train'],loc='best')
import pandas as pd

whole_income_annual = pd.read_csv("../input/19449577-assignment2/whole_income_annual.csv")