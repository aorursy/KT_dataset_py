# Do the below in kaggle

!pip install plotly==4.4.1

!pip install chart_studio

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

_, AMD_PL, AMD_BS, AMD_CF = excel_to_df("/kaggle/input/SimFin-data-AMD.xlsx")
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='19449542dong', api_key='rCTxOlWPZAmKi11mG0cL')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes

# Book Value – Is the shareholder equity reflected in a low price to book value ratio?

# AMD has no preferred stock. Goodwill and other intangible assets are in "Long Term Investments & Receivables" and "Other Long Term Assets"

# Book Value = Total Assets - Intangible assets - Liabilities - Preferred Stock Value(0)

# Calculate book value of AMD From 2010 to 2019

AMD_BS["book value"] = AMD_BS["Total Assets"] - AMD_BS["Other Long Term Assets"] - AMD_BS["Long Term Investments & Receivables"] - AMD_BS["Total Liabilities"] 
# Book Value – Is the shareholder equity reflected in a low price to book value ratio?

# Using Ploty to plot book value , market cap and P/B ratio from 2010 to 2019, to see if the book value is going up or down.

# Or we can see if the book value is negative, if yes, the P/B ratio will be negative too. In this case, The P/B ratio is distorted.





Book_Value_data = go.Bar(

 x=AMD_BS.index,

 y=AMD_BS["book value"],

 name='Book Value') 



#BV_Market_Cap_data = [Market_Cap_data,Book_Value_data ]



layout_Book_Value = go.Layout(

  title ="Book Value " )



fig_AMD_bs_Book_Value = go.Figure(data= Book_Value_data, layout = layout_Book_Value)

fig_AMD_bs_Book_Value.show()

# py.iplot(fig_bs, filename='Total Assets and Book Value')



Market_Cap = [5820, 3930, 1780, 2920,2050,2250,9470,9790,18130, 50300] #In millions, From website: https://www.macrotrends.net/stocks/charts/AMD/amd/market-cap,

Market_Cap_data = go.Bar(

 x=AMD_BS.index,

 y=Market_Cap,

 name='Market Cap')



layout_Market_Cap = go.Layout(

 title ="Market Cap " )



fig_AMD_bs_Market_Cap = go.Figure(data= Market_Cap_data, layout = layout_Market_Cap)

fig_AMD_bs_Market_Cap.show()





AMD_BS["Price_to_Book_Value_ratio"] = Market_Cap / AMD_BS["book value"]



PB_Ratio_data = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["Price_to_Book_Value_ratio"],

 name='Book Value') 



layout_PB_Ratio = go.Layout(

 barmode='stack', title ="Price to Book value Ratio" )



fig_AMD_bs_Book_Value = go.Figure(data=PB_Ratio_data,layout=layout_PB_Ratio)

fig_AMD_bs_Book_Value.show()
# Sales potential – Can the company generate more sales?

# Turn the operating expenses in the PL from negative numbers into positive numbers

AMD_PL["_Operating Expenses"] = -AMD_PL["Operating Expenses"] 



# Using Ploty to plot revenue and operating expenses From 2010 to 2019, to find out what is the trend of the revenue and operating expenses,

# If the revenue and operating expenses are increasing,it may means that the company is gorwing, it may generate more sales.



Revenue_And_Operating_Expenses_data = []

columns = '''

_Operating Expenses

Revenue

'''



for col in columns.strip().split("\n"):

    Revenue_And_Operating_Expenses_Scatter = go.Scatter(

         x=AMD_PL.index,

         y=AMD_PL[ col ],

         name=col) 

     

    Revenue_And_Operating_Expenses_data.append(Revenue_And_Operating_Expenses_Scatter)

 

layout_RE = go.Layout(

    barmode='stack',title ="Revenue and Operating Expenses")



fig_AMD_bs_Sales_potential = go.Figure(data=Revenue_And_Operating_Expenses_data, layout=layout_RE)

fig_AMD_bs_Sales_potential.show()
# Margin pressure – Can the company cut costs or otherwise raise margins?

# Turn the cost of revenue in the PL from negative numbers into positive numbers



AMD_PL["_Cost of revenue"] = - AMD_PL["Cost of revenue"] 



# Using Ploty to plot cost and revenue From 2010 to 2019, to find out what is the trend of the cost and revenue,

# If the revenue increase faster than the cost, it may imply that the company may raise margin.



Cost_And_Revenue_data = []

columns = '''

_Cost of revenue

Revenue

'''



for col in columns.strip().split("\n"):

    Cost_And_Revenue_Scatter = go.Scatter(

         x=AMD_PL.index,

         y=AMD_PL[ col ],

         name=col) 

     

    Cost_And_Revenue_data.append(Cost_And_Revenue_Scatter)

 

layout_CR = go.Layout(

    barmode='stack',title ="Revenue and Cost of Revenue")



fig_AMD_bs_Margin_pressure = go.Figure(data=Cost_And_Revenue_data, layout=layout_CR)

fig_AMD_bs_Margin_pressure.show()



#py.iplot(fig_bs_PR, filename='Cost And Revenue')
# Overheads – Can the company make concrete moves to reduce its expenses?

# Turn the operating expenses in the PL from negative numbers into positive numbers



AMD_PL["_Operating Expenses"] = -AMD_PL["Operating Expenses"] 



#Using Ploty to plot Operating Expenses compared with Total Assets From 2010 to 2019, to see if the percentage of Operating Expenses/Total Assets is going up or down.

# If the the percentage of Operating Expenses/Total Assets is gonig up, it may seems that the company fail to reduce its expenses.



Operating_Expenses_data = go.Scatter(

 x=AMD_PL.index,

 y=AMD_PL["_Operating Expenses"],

 name='Operating Expenses') 

 

Total_Asset_data = go.Bar(

 x=AMD_BS.index,

 y=AMD_BS["Total Assets"],

 name='Total Assets') 



Operating_Expenses_And_Total_Asset_data = [Operating_Expenses_data, Total_Asset_data] 

layout_OE = go.Layout(

 barmode='stack',title ="Operating Expenses and Total_Assets")



fig_AMD_bs_Operating_Expenses = go.Figure(data=Operating_Expenses_And_Total_Asset_data, layout=layout_OE)

fig_AMD_bs_Operating_Expenses.show()

#py.iplot(fig_bs_PR, filename='Operating Expense')



AMD_PL["Operating Expenses/Total Asset"] = AMD_PL["_Operating Expenses"] / AMD_BS["Total Assets"]

AMD_PL[["Operating Expenses/Total Asset"]].plot()
# Costs of sales – What will it cost the company to sell more?

# Find out if there is a posstive correlation between the cost of revenue and revenue, to see if it will take more cost of revenue sell more.

AMD_PL["_Cost of revenue"] = -AMD_PL["Cost of revenue"] 

Cost_of_revenue = AMD_PL["_Cost of revenue"].values

Revenue = AMD_PL["Revenue"].values

get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(Revenue,Cost_of_revenue)

plt.xlabel("Revenue")

plt.ylabel("Cost of revenue")


# Investment plans – How does management plan to reinvest or use the company’s cash on hand?

# Check the change in long term investment, to find out how do the company invest in long term investment.

# Turn the investment data in the PL From negative numbers into positive numbers



AMD_CF["_Change in Fixed Assets & Intangibles"] = -AMD_CF["Change in Fixed Assets & Intangibles"] 

AMD_CF["_Net Change in Long Term Investment"] = -AMD_CF["Net Change in Long Term Investment"] 

AMD_CF["_Net Cash From Acquisitions & Divestitures"] = -AMD_CF["Net Cash From Acquisitions & Divestitures"] 

AMD_CF["_Other Investing Activities"] = -AMD_CF["Other Investing Activities"] 



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

         x=AMD_CF.index,

         y=AMD_CF[ col ],

         name=col ) 

    Investment_data.append(Investment_Scatter)

 

layout_IN = go.Layout(

     barmode='stack',title ="Change in long term investment")



fig_AMD_bs_equity_Investment = go.Figure(data=Investment_data, layout=layout_IN )

fig_AMD_bs_equity_Investment.show()

#py.iplot(fig_bs_equity, filename='Investment Data')
# Insider stock buybacks – Were there clear signs that the company itself or senior executives are buying their own stock?

# Turn the Treasury Stock, Total Equity, Retained Earnings in the BS From negative numbers into positive numbers



AMD_BS["_Treasury Stock"] = - AMD_BS["Treasury Stock"] 



# Using Ploty to plot Treasury Stock,Total Equity,Retained Earnings From 2010 to 2019, to see if the treasury stock is rising. 

# If yes, the company may be buying the outstanding stock back.



Treasury_Stock_data = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["_Treasury Stock"],

 name='Treasury Stock') 

 

Total_Equity_data = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["Total Equity"],

 name='Total Equity') 



Retained_Earnings_data = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["Retained Earnings"],

 name='Retained Earnings') 



Treasury_And_Total_Equity_data = [Total_Equity_data, Treasury_Stock_data,Retained_Earnings_data]

layout_TS = go.Layout(

 barmode='stack',title ="Treasury Stock,Total Equity and Retained Earnings")



fig_AMD_bs_Treasury_Stock = go.Figure(data=Treasury_And_Total_Equity_data, layout=layout_TS)

fig_AMD_bs_Treasury_Stock.show()



#py.iplot(fig_bs_PR, filename='Treasury Stock')
# Liquidation value – What’s the break-up value of the company?

# Using Ploty to plot Book Value, assets, liabilities, and euity from 2010 to 2019, to explain what is the break-up value



assets = go.Bar(

 x=AMD_BS.index,

 y=AMD_BS["Total Assets"],

 name='Assets')



liabilities = go.Bar(

 x=AMD_BS.index,

 y=AMD_BS["Total Liabilities"],

 name='Liabilities')



shareholder_equity = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["Total Equity"],

 name='Equity')



Book_Value_data = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["book value"],

 name='Book Value') 



BVALE_data = [assets, liabilities, shareholder_equity,Book_Value_data ]

layout_BVALE = go.Layout(

 title ="Liquidation value")



fig_AMD_bs_break_up_value = go.Figure(data=BVALE_data, layout=layout_BVALE)

fig_AMD_bs_break_up_value.show()

# py.iplot(fig_AMD_bs_break_up_value, filename='Total Assets and Liabilities')
#Interest Coverage Ratio - Can the company easily repay their interest expenses on outstanding debt?

#6. Calculate EBIT, from 2010 to 2019

AMD_PL["_EBIT"] = AMD_PL["Revenue"] + AMD_PL["Cost of revenue"] - AMD_PL["Operating Expenses"]



#6. Calculate Interest Expenses, From 2010 to 2019

AMD_PL["_Interest Expenses"] = AMD_PL["_EBIT"] - AMD_PL["Pretax Income (Loss)"] 



#6. Calculate Interest Coverage Ratio, From 2010 to 2019

AMD_PL["_Interest Coverage Ratio"] = AMD_PL["_EBIT"] / AMD_PL["_Interest Expenses"] 



#6. Using Ploty to plot Interest Coverage Ratio From 2010 to 2019

Interest_Coverage_Ratio_data = go.Scatter(

 x=AMD_PL.index,

 y=AMD_PL["_Interest Coverage Ratio"],

 name='Interest Coverage Ratio') 



layout_ICR = go.Layout(

 barmode='stack',title ="Interest Coverage Ratio")



fig_AMD_bs_Interest_Coverage_Ratio = go.Figure(data=Interest_Coverage_Ratio_data, layout=layout_ICR)

fig_AMD_bs_Interest_Coverage_Ratio.show()

# py.plot(fig_bs_ICR, filename='Interest Coverage Ratio')
# Additional calculation

#1. Calculate Earnings Per Share (EPS) Annual Compounded Growth Rate, From 2010 to 2019

EPS_2010 = 0.64 #From Website: https://www.macrotrends.net/stocks/charts/AMD/amd/eps-earnings-per-share-diluted

EPS_2019 = 0.30 #From Website: https://www.macrotrends.net/stocks/charts/AMD/amd/eps-earnings-per-share-diluted

Annual_Compounded_Growth_Rate = (EPS_2019/EPS_2010)**(1/9)-1

print("The Annual Compounded Growth Rate of AMD is" , round(Annual_Compounded_Growth_Rate,4)*100,"%")



#2. Estimate EPS 10 years from now 2019

EPS_2029 = EPS_2019*(1+Annual_Compounded_Growth_Rate)**10

print("AMD's Estimate EPS 10 years from now is" ,round(EPS_2029,2))
#3. Determine Current Target Buy Price

Aveerage_PE = 31.65 # Sector:Computer and Technology,Industry:Semiconductor - General,From Website:https://www.macrotrends.net/stocks/sector/10/computer-and-technology 

Future_Stock_Price_10_Years = EPS_2029 * Aveerage_PE

Discount_Rate = 0.0799 # Industry:Semiconductor, From Website: http://people.stern.nyu.edu/adamodar/New_Home_Page/datafile/wacc.htm

Target_Buy_Price_Today = Future_Stock_Price_10_Years / (1 + Discount_Rate )**10

#4.Margin of Safety

Margin_of_Safety = 0.30 # As assumption

Safe_Buy_Price = Target_Buy_Price_Today * (1- Margin_of_Safety)

print("The Safe Buy Price of AMD Today is",round(Safe_Buy_Price,2))
#5.Calculate Debt to Equity Ratio, From 2010 to 2019

AMD_BS["_Debt to Equity Ratio"] = AMD_BS["Total Liabilities"] / AMD_BS["Total Equity"]
#5. Using Ploty to plot Debt to Equity Ratio From 2010 to 2019

Debt_to_Equity_Ratio_data = go.Scatter(

 x=AMD_BS.index,

 y=AMD_BS["_Debt to Equity Ratio"],

 name='Debt to Equity Ratio') 

 

layout_DE = go.Layout(

 barmode='stack',title = "Debt to Equity Ratio")



fig_AMD_bs_Debt_to_Equity_Ratio = go.Figure(data=Debt_to_Equity_Ratio_data, layout=layout_DE)

fig_AMD_bs_Debt_to_Equity_Ratio.show()

# py.plot(fig_AMD_bs_Debt_to_Equity Ratio, filename='Debt to Equity Ratio')