import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')
#import P&L statement from SIMFIN Excel data

PL = pd.read_excel('../input/Costco.xlsx',sheet_name=1,index_col=0)

PL
#plot major P&L items

PL.transpose()[['Revenue','Gross Profit','Operating Income (Loss)',

                'Net Income Available to Common Shareholders']].plot(kind='bar',figsize=(12,6))
#Costco Historical Shares Oustanding data obtained from macrotrends.net

Shares_out = pd.read_html('https://www.macrotrends.net/stocks/charts/COST/costco/shares-outstanding',

                          index_col=0)[0][:10]

Shares_out = Shares_out.sort_index()

Shares_out = Shares_out.rename(index={2010:"FY '10",2011:"FY '11",2012:"FY '12",2013:"FY '13",2014:"FY '14",

                                     2015:"FY '15",2016:"FY '16",2017:"FY '17",2018:"FY '18",2019:"FY '19"})

Shares_out['Costco Annual Shares Outstanding(Millions of Shares).1']
#Calculate and plot the Costco EPS from 2010 - 2019

Net_income = PL.transpose()['Net Income Available to Common Shareholders']

Shares_outstanding = Shares_out['Costco Annual Shares Outstanding(Millions of Shares).1']

EPS = Net_income.divide(Shares_outstanding)



EPS.plot(title='Earnings Per Share',figsize=(8,6))
#Calculate the EPS CAGR

EPS_CAGR = (((EPS["FY '19"]/EPS["FY '10"]) ** (1/9)) - 1) * 100

print('EPS CAGR: ' + str(round((EPS_CAGR),2)) + '%')



#Calculate PEG Ratio

PE_Ratio = 36.97  # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/217867

PEG_Ratio = PE_Ratio/EPS_CAGR

print('PEG Ratio: ' + str(round((PEG_Ratio),2)))
#Calculate Estimated Future EPS 10 years from now

Future_EPS = EPS["FY '19"] * ((1 + (EPS_CAGR/100)) ** 10)

print('Estimated Future EPS: ' + str(round((Future_EPS),2)))
#Obtain Costco Average PE Ratio from 'https://www.macrotrends.net/stocks/charts/COST/costco/pe-ratio'

pe = pd.read_html('https://www.macrotrends.net/stocks/charts/COST/costco/pe-ratio',index_col=0)[0]

pe = pe['Costco PE Ratio Historical Data']['PE Ratio'].loc['2019-11-30':'2009-11-30']

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

Current_Price = pd.read_html('https://finance.yahoo.com/quote/COST?p=COST',index_col=0)[0].loc['Previous Close'][1]



print('Target Buy Price: ' + str(round((Target_Buy_Price),2)))

print('Adjusted Target Price: ' + str(round((Adjusted_Target_Price),2)))

print('Current Price: ' + str(Current_Price))
#Import Costco Balance Sheet from SIMFIN Excel Data

BS = pd.read_excel('../input/Costco.xlsx',sheet_name=2,index_col=0)



#plot the Major Balance Sheet Items

BS.transpose()[['Total Assets','Total Liabilities','Total Equity']].plot(kind='bar',figsize=(12,6))
#Calculate and Plot the Debt to Equity Ratio

Debt = BS.transpose()['Long Term Debt']

Equity = BS.transpose()['Total Equity']

DE_Ratio = Debt.divide(Equity) * 100

DE_Ratio.plot(title='Debt to Equity Ratio (%)',figsize=(8,6))

print (DE_Ratio)
#Calculate & Plot the Interest Coverage Ratio

EBIT = PL.transpose()['Operating Income (Loss)']

Net_Interest = PL.transpose()['Non-Operating Income (Loss)'] * -1



Net_Interest_Cover = EBIT.divide(Net_Interest).apply(lambda x: np.nan if x <= 0 else x)



"""Note: Interest cover is infinite and therefore not a number (nan) when there is no 

interest or when net interest is positive, such as in years 2012 & 2019"""



print('Net Interest Coverage Ratio')

print(Net_Interest_Cover)