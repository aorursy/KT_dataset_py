#!pip install plotly==4.4.1
#!pip install chart_studio
#!pip install xlrd 
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
df, Walmart_PL, Walmart_BS, Walmart_CF = excel_to_df("/kaggle/input/Walmart-data.xlsx")
Walmart_BS
del(Walmart_BS["Assets"])

Walmart_BS
Walmart_BS["_Total Current Assets"] = Walmart_BS["Cash, Cash Equivalents & Short Term Investments"] + Walmart_BS["Accounts & Notes Receivable"] + Walmart_BS["Inventories"] + Walmart_BS["Other Short Term Assets"]
Walmart_BS[["_Total Current Assets", "Total Current Assets"]]
Walmart_BS["_NonCurrent Assets"] = Walmart_BS["Property, Plant & Equipment, Net"] + Walmart_BS["Other Long Term Assets"]
Walmart_BS["_Total Assets"] = Walmart_BS["_NonCurrent Assets"] + Walmart_BS["_Total Current Assets"] 
Walmart_BS["_Total Liabilities"] = Walmart_BS["Total Current Liabilities"] + Walmart_BS["Total Noncurrent Liabilities"]
Walmart_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline
Walmart_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
Walmart_BS[ asset_columns ].plot()
#!pip install chart_studio
#!pip install xlrd
#!pip install plotly
import chart_studio
chart_studio.tools.set_credentials_file(username='cg5125', api_key='u9s9fxP5XAcytbbEMPuL')
import chart_studio.plotly as py
import plotly.graph_objs as go

assets = go.Bar(
    x=Walmart_BS.index,
    y=Walmart_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=Walmart_BS.index,
    y=Walmart_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=Walmart_BS.index,
    y=Walmart_BS["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
fig_bs.show()
py.plot(fig_bs, filename='Total Assets and Liabilities')
asset_data = []
columns = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=Walmart_BS.index,
        y=Walmart_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
fig_bs_assets.show()
py.plot(fig_bs_assets, filename='Total Assets Breakdown')
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
        x=Walmart_BS.index,
        y=Walmart_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
fig_bs_liabilitys.show()
py.plot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
Walmart_BS["working capital"] = Walmart_BS["Total Current Assets"] - Walmart_BS["Total Current Liabilities"]
Walmart_BS[["working capital"]].plot()
Walmart_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 

PR_data = []
columns = '''
Accounts & Notes Receivable
Payables & Accruals
'''

for col in columns.strip().split("\n"):
    PR_Scatter = go.Scatter(
        x=Walmart_BS.index,
        y=Walmart_BS[ col ],
        name=col
    )    
    PR_data.append(PR_Scatter)
    
layout_PR = go.Layout(
    barmode='stack'
)

fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)
fig_bs_PR.show()
py.plot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
Walmart_BS["Inventories"].plot()
Walmart_BS[ ["Property, Plant & Equipment, Net",  "Other Long Term Assets"] ].plot()
# Using Plotly

AAA_data = []
columns = '''
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    AAA_bar = go.Bar(
        x=Walmart_BS.index,
        y=Walmart_BS[ col ],
        name=col
    )    
    AAA_data.append(AAA_bar)
    
layout_AAA = go.Layout(
    barmode='stack'
)

fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)
fig_bs_AAA.show()
py.plot(fig_bs_AAA, filename='Total Long Term Assets')
equity_columns = '''
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
Minority Interest
'''

equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
Walmart_BS[ equity_columns ].plot()
# Using Plotly

equity_data = []
columns = '''
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
Minority Interest
'''


for col in columns.strip().split("\n"):
    equity_Scatter = go.Scatter(
        x=Walmart_BS.index,
        y=Walmart_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)
fig_bs_equity.show()
py.plot(fig_bs_equity, filename='Total Equity')
Walmart_BS["book value"] = Walmart_BS["Total Assets"] - Walmart_BS["Total Liabilities"]

Walmart_BS["book value"].plot()
Walmart_BS["current ratio"] = Walmart_BS["Total Current Assets"] / Walmart_BS["Total Current Liabilities"]
print(Walmart_BS["current ratio"].values)
Walmart_BS["current ratio"].plot()
PE_RATIO = 22.06 # FROM WEBSITE: https://www.macrotrends.net/stocks/charts/WMT/walmart/pe-ratio

GROWTH_RATE = 0.20166 # Forcast over the five next years

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("Walmart Wholesale's PEG Ratio is", PEG_ratio)
#End of Value Investing Stock Analysis Template
import plotly.graph_objs as go
import plotly.offline as pltoff
AR=Walmart_PL[["Revenue"]]
ARG_2010=((AR.values[1][0])-(AR.values[0][0]))/(AR.values[0][0])*100
ARG_2011=((AR.values[2][0])-(AR.values[1][0]))/(AR.values[1][0])*100
ARG_2012=((AR.values[3][0])-(AR.values[2][0]))/(AR.values[2][0])*100
ARG_2013=((AR.values[4][0])-(AR.values[3][0]))/(AR.values[3][0])*100
ARG_2014=((AR.values[5][0])-(AR.values[4][0]))/(AR.values[4][0])*100
ARG_2015=((AR.values[6][0])-(AR.values[5][0]))/(AR.values[5][0])*100
ARG_2016=((AR.values[7][0])-(AR.values[6][0]))/(AR.values[6][0])*100
ARG_2017=((AR.values[8][0])-(AR.values[7][0]))/(AR.values[7][0])*100
ARG_2018=((AR.values[9][0])-(AR.values[8][0]))/(AR.values[8][0])*100
print('Annual Revenue Growth='+format(ARG_2010,'.2f')+'%,'+format(ARG_2011,'.2f')+'%,'+format(ARG_2012,'.2f')+'%,'+format(ARG_2013,'.2f')+'%,'+format(ARG_2014,'.2f')+'%,'+format(ARG_2015,'.2f')+'%,'+format(ARG_2016,'.2f')+'%,'+format(ARG_2017,'.2f')+'%,'+format(ARG_2018,'.2f')+'%,')
ARG=[[ARG_2010,'2010'],[ARG_2011,'2011'],[ARG_2012,'2012'],[ARG_2013,'2013'],[ARG_2014,'2014'],[ARG_2015,'2015'],[ARG_2016,'2016'],[ARG_2017,'2017'],[ARG_2018,'2018']]
print('The year which Annual Revenue Growth > 5% are:')
for i in range(len(ARG)-1):
    if ARG[i][0]>5:
        print(ARG[i][1])
        
AP=Walmart_PL[["Gross Profit"]]
APG_2010=((AP.values[1][0])-(AP.values[0][0]))/(AP.values[0][0])*100
APG_2011=((AP.values[2][0])-(AP.values[1][0]))/(AP.values[1][0])*100
APG_2012=((AP.values[3][0])-(AP.values[2][0]))/(AP.values[2][0])*100
APG_2013=((AP.values[4][0])-(AP.values[3][0]))/(AP.values[3][0])*100
APG_2014=((AP.values[5][0])-(AP.values[4][0]))/(AP.values[4][0])*100
APG_2015=((AP.values[6][0])-(AP.values[5][0]))/(AP.values[5][0])*100
APG_2016=((AP.values[7][0])-(AP.values[6][0]))/(AP.values[6][0])*100
APG_2017=((AP.values[8][0])-(AP.values[7][0]))/(AP.values[7][0])*100
APG_2018=((AP.values[9][0])-(AP.values[8][0]))/(AP.values[8][0])*100
print('Annual Profit Growth='+format(APG_2010,'.2f')+'%,'+format(APG_2011,'.2f')+'%,'+format(APG_2012,'.2f')+'%,'+format(APG_2013,'.2f')+'%,'+format(APG_2014,'.2f')+'%,'+format(APG_2015,'.2f')+'%,'+format(APG_2016,'.2f')+'%,'+format(APG_2017,'.2f')+'%,'+format(APG_2018,'.2f')+'%,')
APG=[[APG_2010,'2010'],[APG_2011,'2011'],[APG_2012,'2012'],[APG_2013,'2013'],[APG_2014,'2014'],[APG_2015,'2015'],[APG_2016,'2016'],[APG_2017,'2017'],[APG_2018,'2018'],]
print('The year which Annual Profit Growth > 5% are:')
for i in range(len(APG)-1):
    if APG[i][0]>5:
        print(APG[i][1])

Walmart_CF[["Cash from Operating Activities"]]
CA=Walmart_BS[["Total Current Assets"]]
CL=Walmart_BS[["Total Current Liabilities"]]
CR_2009=(CA.values[0][0])/(CL.values[0][0])
CR_2010=(CA.values[1][0])/(CL.values[1][0])
CR_2011=(CA.values[2][0])/(CL.values[2][0])
CR_2012=(CA.values[3][0])/(CL.values[3][0])
CR_2013=(CA.values[4][0])/(CL.values[4][0])
CR_2014=(CA.values[5][0])/(CL.values[5][0])
CR_2015=(CA.values[6][0])/(CL.values[6][0])
CR_2016=(CA.values[7][0])/(CL.values[7][0])
CR_2017=(CA.values[8][0])/(CL.values[8][0])
CR_2018=(CA.values[9][0])/(CL.values[9][0])
print('Current Ratio='+format(CR_2009,'.2f')+','+format(CR_2010,'.2f')+','+format(CR_2011,'.2f')+','+format(CR_2012,'.2f')+','+format(CR_2013,'.2f')+','+format(CR_2014,'.2f')+','+format(CR_2015,'.2f')+','+format(CR_2016,'.2f')+','+format(CR_2017,'.2f')+','+format(CR_2018,'.2f')+',')
CR=[[CR_2009,'2009'],[CR_2010,'2010'],[CR_2011,'2011'],[CR_2012,'2012'],[CR_2013,'2013'],[CR_2014,'2014'],[CR_2015,'2015'],[CR_2016,'2016'],[CR_2017,'2017'],[CR_2018,'2018']]
print('The year which Current Ratio > 1 are:')
for i in range(len(ARG)-1):
    if ARG[i][0]>1:
        print(ARG[i][1])
#Additional Calculation
#1. Earnings Per Share(EPS)Annual Compounded Growth Rate
#2. Estimate EPS 10 years from now
#3. Determine Current Target Buy Price
#4. Margin of Safety(25% off the target buy price)
#5. Debt to Equity Ratio
#6. Interest COverage Ratio

Total_shares = 2868000000
#data from https://www.nasdaq.com/market-activity/stocks/wmt/institutional-holdings
EPS=Walmart_PL[["Net Income Available to Common Shareholders"]]*1000000/Total_shares
print(EPS)
CAGR=(pow((EPS.values[9][0])/(EPS.values[0][0]),1/9)-1)
print('Annual Compounded Growth Rate='+format(CAGR*100,'.2f')+'%')
Future_EPS=(EPS.values[9][0])*((CAGR+1)**9)
print('EPS 10 Years from now='+format(Future_EPS,'.2f'))
Stockprice_2009=50.55
Stockprice_2010=53.00
Stockprice_2011=54.35
Stockprice_2012=67.22
Stockprice_2013=75.32
Stockprice_2014=77.33
Stockprice_2015=72.49
Stockprice_2016=69.55
Stockprice_2017=78.96
Stockprice_2018=92.37
#data from https://www.macrotrends.net/stocks/charts/WMT/walmart/stock-price-history
PE_Ratio_2009=Stockprice_2009/(EPS.values[0][0])
PE_Ratio_2010=Stockprice_2010/(EPS.values[1][0])
PE_Ratio_2011=Stockprice_2011/(EPS.values[2][0])
PE_Ratio_2012=Stockprice_2012/(EPS.values[3][0])
PE_Ratio_2013=Stockprice_2013/(EPS.values[4][0])
PE_Ratio_2014=Stockprice_2014/(EPS.values[5][0])
PE_Ratio_2015=Stockprice_2015/(EPS.values[6][0])
PE_Ratio_2016=Stockprice_2016/(EPS.values[7][0])
PE_Ratio_2017=Stockprice_2017/(EPS.values[8][0])
PE_Ratio_2018=Stockprice_2018/(EPS.values[9][0])
#Caculate PE Ratio from 2009 to 2018
PE_Ratio=[PE_Ratio_2009,PE_Ratio_2010,PE_Ratio_2011,PE_Ratio_2012,PE_Ratio_2013,PE_Ratio_2014,PE_Ratio_2015,PE_Ratio_2016,PE_Ratio_2017,PE_Ratio_2018]
average_PE=np.mean(PE_Ratio)
print(PE_Ratio)
print('Average PE Ratio='+format(average_PE,'.2f'))
Future_stockprice=average_PE*Future_EPS
print('Stock price 10 years from now='+format(Future_stockprice,'.2f'))
Discount_Rate=0.075
Target_buyprice_today=Future_stockprice/((1+Discount_Rate)**9)
print('Target buy price today='+format(Target_buyprice_today,'.2f'))
margin=0.25
margin_safety=Target_buyprice_today*(1-margin)
print('Margin of safety='+format(margin_safety,'.2f'))
Debt=Walmart_BS[["Total Liabilities"]]
Equity=Walmart_BS[["Total Equity"]]
print(Debt)
print(Equity)
DER_2009=(Debt.values[0][0])/(Equity.values[0][0])
DER_2010=(Debt.values[1][0])/(Equity.values[1][0])
DER_2011=(Debt.values[2][0])/(Equity.values[2][0])
DER_2012=(Debt.values[3][0])/(Equity.values[3][0])
DER_2013=(Debt.values[4][0])/(Equity.values[4][0])
DER_2014=(Debt.values[5][0])/(Equity.values[5][0])
DER_2015=(Debt.values[6][0])/(Equity.values[6][0])
DER_2016=(Debt.values[7][0])/(Equity.values[7][0])
DER_2017=(Debt.values[8][0])/(Equity.values[8][0])
DER_2018=(Debt.values[9][0])/(Equity.values[9][0])
#Caculate Debt to Equity Ratio from 2000 to 2018
DER_Ratio=[DER_2009,DER_2010,DER_2011,DER_2012,DER_2013,DER_2014,DER_2015,DER_2016,DER_2017,DER_2018]
average_DER=np.mean(DER_Ratio)
print(DER_Ratio)
print('Average Debt to Equity Ratio='+format(average_DER,'.2f'))
EBIT=Walmart_PL[["Operating Income (Loss)"]]*1000000
print(EBIT)
interest_expense_2018=2330000000
interest_expense_2017=2367000000
interest_expense_2016=2548000000
interest_expense_2015=2461000000
interest_expense_2014=2335000000
interest_expense_2013=2249000000
interest_expense_2012=2320000000
interest_expense_2011=2205000000
interest_expense_2010=2065000000
interest_expense_2009=2184000000
#data from http://basic.10jqka.com.cn/WMT/finance.html
ICR_2009=(EBIT.values[0][0])/interest_expense_2009
ICR_2010=(EBIT.values[1][0])/interest_expense_2010
ICR_2011=(EBIT.values[2][0])/interest_expense_2011
ICR_2012=(EBIT.values[3][0])/interest_expense_2012
ICR_2013=(EBIT.values[4][0])/interest_expense_2013
ICR_2014=(EBIT.values[5][0])/interest_expense_2014
ICR_2015=(EBIT.values[6][0])/interest_expense_2015
ICR_2016=(EBIT.values[7][0])/interest_expense_2016
ICR_2017=(EBIT.values[8][0])/interest_expense_2017
ICR_2018=(EBIT.values[9][0])/interest_expense_2018
#Caculate Interest Coverage Ratio from 2009 to 2018
ICR_Ratio=[ICR_2009,ICR_2010,ICR_2011,ICR_2012,ICR_2013,ICR_2014,ICR_2015,ICR_2016,ICR_2017,ICR_2018]
average_ICR=np.mean(ICR_Ratio)
print('Average Interest Coverage Ratio='+format(average_ICR,'.2f'))