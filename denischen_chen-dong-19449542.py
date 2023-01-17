# Do the below in kaggle

#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe



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
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'

# Look for it under usr/bin on the right drawer



# import excel_to_df function

import os

import matplotlib.pyplot as plt

#import tk_library_py

#from tk_library_py import excel_to_df
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, NVDA_PL, NVDA_BS, NVDA_CF = excel_to_df("/kaggle/input/simfindatanvda/SimFin-data - NVDA.xlsx")
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='Denis_Chen', api_key='F9gT67qcSvZLsGXQybej')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes

# 1. Calculate Earnings Per Share (EPS) Annual Compounded Growth Rate, From 2011 to 2020

EPS_2011 = 0.43     #From Website: https://www.macrotrends.net/stocks/charts/NVDA/nvidia/eps-earnings-per-share-diluted

EPS_2020 = 4.52     #From Website: https://www.macrotrends.net/stocks/charts/NVDA/nvidia/eps-earnings-per-share-diluted

Annual_Compounded_Growth_Rate = (EPS_2020/EPS_2011)**(1/9)-1

print("The Annual Compounded Growth Rate of NVIDIA is" , round(Annual_Compounded_Growth_Rate,4)*100,"%")



# 2. Estimate EPS 10 years from now 2020

EPS_2030 = EPS_2020*(1+Annual_Compounded_Growth_Rate)**10

print("NVIDIA's Estimate EPS 10 years from now is" ,round(EPS_2030,2))



# 3. Determine Current Target Buy Price

Aveerage_PE = 31.65     # Sector:Computer and Technology,Industry:Semiconductor - General,From Website:https://www.macrotrends.net/stocks/sector/10/computer-and-technology 

Future_Stock_Price_10_Years = EPS_2030 * Aveerage_PE

Discount_Rate = 0.0799  # Industry:Semiconductor, From Website: http://people.stern.nyu.edu/adamodar/New_Home_Page/datafile/wacc.htm

Target_Buy_Price_Today = Future_Stock_Price_10_Years / (1 + Discount_Rate )**10



# 4. Margin of Safety

Margin_of_Safety = 0.25    # As assumption

Safe_Buy_Price = Target_Buy_Price_Today * (1- Margin_of_Safety)

print("The Safe Buy Price of NVIDIA Today is",round(Safe_Buy_Price,2))
#5.Calculate Debt to Equity Ratio, From 2011 to 2020

NVDA_BS["_Debt to Equity Ratio"] = NVDA_BS["Total Liabilities"] / NVDA_BS["Total Equity"]
#5. Using Ploty to plot Debt to Equity Ratio From 2009 to 2018

Debt_to_Equity_Ratio_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["_Debt to Equity Ratio"],

        name='Debt to Equity Ratio')   

    

layout_DE = go.Layout(

    barmode='stack',

    title = "Debt to Equity Ratio")



fig_NVDA_bs_Debt_to_Equity_Ratio = go.Figure(data=Debt_to_Equity_Ratio_data, layout=layout_DE)

fig_NVDA_bs_Debt_to_Equity_Ratio.show()

# py.iplot(fig_bs_PR, filename='Debt to Equity Ratio')

#6. Calculate EBIT, From 2011 to 2020

NVDA_PL["_EBIT"] = NVDA_PL["Revenue"] + NVDA_PL["Cost of revenue"] - NVDA_PL["Operating Expenses"]
#6. Calculate Interest Expenses, From 2011 to 2020

NVDA_PL["_Interest Expenses"] = NVDA_PL["_EBIT"] - NVDA_PL["Pretax Income (Loss)"] 
#6. Calculate Interest Coverage Ratio, From 2011 to 2020

NVDA_PL["_Interest Coverage Ratio"] = NVDA_PL["_EBIT"] / NVDA_PL["_Interest Expenses"] 
#6. Using Ploty to plot Interest Coverage Ratio From 2009 to 2018

Interest_Coverage_Ratio_data = go.Scatter(

        x=NVDA_PL.index,

        y=NVDA_PL["_Interest Coverage Ratio"],

        name='Interest Coverage Ratio')   



layout_IRC = go.Layout(

    barmode='stack',

    title = "Interest Coverage Ratio")



fig_NVDA_bs_Interest_Coverage_Ratio = go.Figure(data=Interest_Coverage_Ratio_data, layout=layout_IRC)

fig_NVDA_bs_Interest_Coverage_Ratio.show()

# py.iplot(fig_bs_PR, filename='Interest Coverage Ratio')
# Book Value – Is the shareholder equity reflected in a low price to book value ratio?

# NVIDIA has no preferred stock

# BV = Total Assets - Intangible assets - Liabilities - Preferred Stock Value

# Calculate book value of NVIDIA From 2009 to 2018

NVDA_BS["book value"] = NVDA_BS["Total Assets"] - NVDA_BS["Other Long Term Assets"] -NVDA_BS["Total Liabilities"]
# Book Value – Is the shareholder equity reflected in a low price to book value ratio?

#Using Ploty to plot Book Value From 2009 to 2018, to see if the book value is going up or down.



Book_Value_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["book value"],

        name='Book Value')  



assets = go.Bar(

    x=NVDA_BS.index,

    y=NVDA_BS["Total Assets"],

    name='Assets')



BV_Assets_data = [assets,Book_Value_data ]



layout_BV_Assets = go.Layout(

    barmode='stack', 

    title= 'Total Assets and Book Value')



fig_NVDA_bs_Book_Value = go.Figure(data=BV_Assets_data, layout=layout_BV_Assets)

fig_NVDA_bs_Book_Value.show()

#py.plot(fig_NVDA_bs_Book_Value, filename='Total Assets and Book Value')
# Margin pressure – Can the company cut costs or otherwise raise margins?

# Turn the Cost of revenue in the PL From negative numbers into positive numbers





NVDA_PL["_Cost of revenue"] = -NVDA_PL["Cost of revenue"] 



# Using Ploty to plot Cost And Revenue From 2009 to 2018,to find out what is the trend of the cost and revenue,

# If the revenue increase faster than the cost, it may imply that the company may raise margin.



Cost_And_Revenue_data = []

columns = '''

_Cost of revenue

Revenue

'''



for col in columns.strip().split("\n"):

    Cost_And_Revenue_Scatter = go.Scatter(

        x=NVDA_PL.index,

        y=NVDA_PL[ col ],

        name=col)   

    

    Cost_And_Revenue_data.append(Cost_And_Revenue_Scatter)

    

layout_CR = go.Layout(

    barmode='stack',

    title ="Revenue VS Cost of Revenue")



fig_NVDA_bs_Margin_pressure = go.Figure(data=Cost_And_Revenue_data, layout=layout_CR)

fig_NVDA_bs_Margin_pressure.show()

# py.iplot(fig_bs_PR, filename='Cost And Revenue')
# Overheads – Can the company make concrete moves to reduce its expenses?

# Turn the Operating Expenses in the PL From negative numbers into positive numbers



NVDA_PL["_Operating Expenses"] = -NVDA_PL["Operating Expenses"] 



#Using Ploty to plot Operating Expenses from 2009 to 2018,to see if  Operating Expenses increase faster than Total Assets.

# If the Operating Expenses increase faster, it may seems that the company fail to reduce its expenses.



Operating_Expenses_data = go.Scatter(

        x=NVDA_PL.index,

        y=NVDA_PL["_Operating Expenses"],

        name='Operating Expenses')   

    

Total_Asset_data = go.Bar(

        x=NVDA_BS.index,

        y=NVDA_BS["Total Assets"],

        name='Total Assets') 



Operating_Expenses_And_Total_Asset_data = [Operating_Expenses_data, Total_Asset_data]    

layout_OE = go.Layout(

    barmode='stack',

    title ="Operating Expenses VS Total Assets")



fig_NVDA_bs_Operating_Expenses = go.Figure(data=Operating_Expenses_And_Total_Asset_data, layout=layout_OE)

fig_NVDA_bs_Operating_Expenses.show()

# py.iplot(fig_bs_PR, filename='Operating Expense')
# Costs of sales – What will it cost the company to sell more?

# Find out if there is a posstive correlation between the cost of revenue and revenue, to see if it will take more cost of revenue sell more.

NVDA_PL["_Cost of revenue"] = -NVDA_PL["Cost of revenue"] 

Cost_of_revenue = NVDA_PL["_Cost of revenue"].values

Revenue = NVDA_PL["Revenue"].values

get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(Revenue,Cost_of_revenue)

plt.xlabel("Revenue")

plt.ylabel("Cost of revenue")

plt.title('Relationship betwwen cost of revenue')


# Investment plans – How does management plan to reinvest or use the company’s cash on hand?

# Check the change in long term investment, to find out how do the company invest in long term investment.



# Turn the Investment Data in the PL From negative numbers into positive numbers

NVDA_CF["_Change in Fixed Assets & Intangibles"] = -NVDA_CF["Change in Fixed Assets & Intangibles"] 

NVDA_CF["_Net Change in Long Term Investment"] = -NVDA_CF["Net Change in Long Term Investment"] 

NVDA_CF["_Net Cash From Acquisitions & Divestitures"] = -NVDA_CF["Net Cash From Acquisitions & Divestitures"] 

NVDA_CF["_Other Investing Activities"] = -NVDA_CF["Other Investing Activities"] 



# Using Plotly to plot Investment Data

Investment_data = []

columns = '''

_Change in Fixed Assets & Intangibles

_Net Change in Long Term Investment

_Net Cash From Acquisitions & Divestitures

_Other Investing Activities

'''





for col in columns.strip().split("\n"):

    Investment_Scatter = go.Scatter(

        x=NVDA_CF.index,

        y=NVDA_CF[ col ],

        name=col    )    

    Investment_data.append(Investment_Scatter)

    

layout_IN = go.Layout(

    barmode='stack',

    title ="Change in long term investment")



fig_NVDA_bs_equity_Investment = go.Figure(data=Investment_data, layout=layout_IN )

fig_NVDA_bs_equity_Investment.show()

# py.iplot(fig_bs_equity, filename='Investment Data')
#Insider stock buybacks – Were there clear signs that the company itself or senior executives are buying their own stock?



# Turn the Treasury Stock, Total Equity, Retained Earnings in the BS From negative numbers into positive numbers



NVDA_BS["_Treasury Stock"] = -NVDA_BS["Treasury Stock"] 



#Using Ploty to plot Treasury Stock,Total Equity,RetainedEarnings From 2009 to 2018,to see if the treasury stock is rising. 

# If yes, the company may be buying the outstanding stock back.



Treasury_Stock_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["_Treasury Stock"],

        name='Treasury Stock')   

    

Total_Equity_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["Total Equity"],

        name='Total Equity') 



Retained_Earnings_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["Retained Earnings"],

        name='Retained Earnings') 



Treasury_And_Total_Equity_data = [Total_Equity_data, Treasury_Stock_data,Retained_Earnings_data]

layout_TS = go.Layout(

    barmode='stack', 

    title="Treasury Stock,Total Equity and Retained Earnings")



fig_NVDA_bs_Treasury_Stock = go.Figure(data=Treasury_And_Total_Equity_data, layout=layout_TS)

fig_NVDA_bs_Treasury_Stock.show()



# py.iplot(fig_bs_PR, filename='Treasury Stock')
# Liquidation value – What’s the break-up value of the company?

# Using Ploty to plot Book Value, assets, liabilities, and euity From 2009 to 2018, to explain the break-up value



assets = go.Bar(

    x=NVDA_BS.index,

    y=NVDA_BS["Total Assets"],

    name='Assets')



liabilities = go.Bar(

    x=NVDA_BS.index,

    y=NVDA_BS["Total Liabilities"],

    name='Liabilities')



shareholder_equity = go.Scatter(

    x=NVDA_BS.index,

    y=NVDA_BS["Total Equity"],

    name='Equity')



Book_Value_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["book value"],

        name='Book Value')  



BVALE_data = [assets, liabilities, shareholder_equity,Book_Value_data ]

layout_BVALE = go.Layout(

    barmode='stack',

    title ="Liquidation value")



fig_NVDA_bs_break_up_value = go.Figure(data=BVALE_data, layout=layout_BVALE)

fig_NVDA_bs_break_up_value.show()

#py.iplot(fig_bs, filename='Total Assets and Liabilities')
# Current Ratio

#current ratio = current asset / current liabilties



NVDA_BS["current ratio"] = NVDA_BS["Total Current Assets"] / NVDA_BS["Total Current Liabilities"]
#Using Ploty to plot Current Ratio From 2009 to 2018



Current_Ratio_data = go.Scatter(

        x=NVDA_BS.index,

        y=NVDA_BS["current ratio"],

        name='Current Ratio')   

    

layout_CR = go.Layout(

    barmode='stack',

    title = "Current Ratio")



fig_NVDA_bs_Current_Ratio = go.Figure(data=Current_Ratio_data, layout=layout_CR)

fig_NVDA_bs_Current_Ratio.show()

# py.iplot(fig_bs_PR, filename='Current Ratio')