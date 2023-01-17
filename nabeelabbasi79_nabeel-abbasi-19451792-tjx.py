import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')
#import P&L statement from SIMFIN Excel data

PL = pd.read_excel('../input/TJX.xlsx',sheet_name=1,index_col=0)

PL.dropna()
#plot major P&L items

PL.transpose()[['Revenue','Gross Profit','Operating Income (Loss)',

                'Net Income Available to Common Shareholders']].plot(kind='bar',figsize=(12,6))
#import Balance Sheet from SIMFIN Excel data

BS = pd.read_excel('../input/TJX.xlsx',sheet_name=2,index_col=0)

BS.dropna()
# Plot Major Balance Sheet Items

BS.transpose()[['Total Assets','Total Liabilities','Total Equity']].plot(kind='bar',figsize=(12,6))
#import Cashflow Statement from SIMFIN Excel data

CF = pd.read_excel('../input/TJX.xlsx',sheet_name=3,index_col=0)

CF.dropna()
# Plot Major Cashflow Statement Items

CF.transpose()[['Cash from Operating Activities','Cash from Investing Activities',

                'Cash from Financing Activities']].plot(kind='bar',figsize=(12,6))
DE_Ratio = BS.transpose()['Long Term Debt']/BS.transpose()['Total Equity'] * 100

DE_Ratio.plot(title='Debt to Equity Ratio (%)')
interest_cover = (PL.transpose()['Operating Income (Loss)'] / PL.transpose()['Non-Operating Income (Loss)']) * -1

interest_cover.plot(title='Interest Coverage Ratio')
ratios = pd.read_excel('../input/TJX.xlsx',sheet_name=4,index_col=0)

ratios
#TJX Historical Shares Oustanding data obtained from macrotrends.net

Shares_out = pd.read_html('https://www.macrotrends.net/stocks/charts/TJX/tjx/shares-outstanding',

                          index_col=0)[0][2:12]

Shares_out = Shares_out.sort_index()

Shares_out = Shares_out.rename(index={2009:"FY '09",2010:"FY '10",2011:"FY '11",2012:"FY '12",2013:"FY '13"

                                      ,2014:"FY '14",2015:"FY '15",2016:"FY '16",2017:"FY '17",2018:"FY '18"})

Shares_out['TJX Annual Shares Outstanding(Millions of Shares).1']
#Calculate and plot the TJX EPS from 2009 - 2018

Net_income = PL.transpose()['Net Income Available to Common Shareholders']

Shares_outstanding = Shares_out['TJX Annual Shares Outstanding(Millions of Shares).1']

EPS = Net_income.divide(Shares_outstanding)



EPS.plot(title='Earnings Per Share')
#Calculate the EPS CAGR

EPS_CAGR = (((EPS["FY '18"]/EPS["FY '09"]) ** (1/9)) - 1) * 100

print('EPS CAGR: ' + str(round((EPS_CAGR),2)) + '%')



#Calculate PEG Ratio

PE_Ratio = ratios.loc['Price to Earnings Ratio (TTM)'][0]

PEG_Ratio = PE_Ratio/EPS_CAGR

print('PEG Ratio: ' + str(round((PEG_Ratio),2)))
#Calculate Estimated Future EPS 10 years from now

EPS_2019 = ratios.loc['Basic EPS (TTM)'][0]

Future_EPS = EPS_2019 * ((1 + (EPS_CAGR/100)) ** 10)

print('Estimated Future EPS: ' + str(round((Future_EPS),2)))
#Obtain TJX Average PE Ratio from 'https://www.macrotrends.net/stocks/charts/TJX/tjx/pe-ratio'

pe = pd.read_html('https://www.macrotrends.net/stocks/charts/TJX/tjx/pe-ratio',index_col=0)[0]

pe = pe['TJX PE Ratio Historical Data']['PE Ratio']

Average_PE_Ratio = pe.mean()

print('Average PE Ratio: ' + str(round((Average_PE_Ratio),2)))



#Calculate the Estimated Future Stock Price 10 years from now

Future_Price = Future_EPS * Average_PE_Ratio

print('Estimated Future Price: ' + str(round((Future_Price),2)))
#Calculate Target Buy Price

Discount_Rate = 0.08 #Median WACC for consumer staples sector. Source: Finbox.com

Margin_of_safety = 0.25



Target_Buy_Price = Future_Price * ((1 + Discount_Rate) ** -10)

Adjusted_Target_Price = Target_Buy_Price * (1 - Margin_of_safety)

Current_Price = pd.read_html('https://finance.yahoo.com/quote/TJX?p=TJX&.tsrc=fin-srch',

                             index_col=0)[0].loc['Previous Close'][1]



print('Target Buy Price: ' + str(round((Target_Buy_Price),2)))

print('Adjusted Target Price: ' + str(round((Adjusted_Target_Price),2)))

print('Current Price: ' + str(Current_Price))