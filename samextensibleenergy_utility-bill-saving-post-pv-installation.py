import os

print(os.listdir("../input"))



# Exploratory analysis tools

import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

from string import ascii_letters

import time



# Data Visualization Tools

import seaborn as sns

from matplotlib import pyplot

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.tools as tls
raw_data = pd.read_csv('../input/Data_Science_Sample_Case_Data.csv')

raw_df = raw_data # making a base copy of the raw data

raw_df = raw_df.drop(['Unnamed: 0'], axis=1) # dropping the index axis for simplicity
# Renaming the column A and column B as respectively below. The colar production could be identified fromt he pattern of the data below.

raw_df = raw_df.rename(index=str, columns={"Column A": "power_demand_kW", "Column B": "solar_kW"}) 

#raw_df.head()
# preparing dataframe for billing calcultions:

#   --- Add a timestamp column 

#   --- Add a weekday colum to easily notify weekends for peak hour separation



df = raw_df

df['TimeStamp'] = pd.to_datetime(df[['Day', 'Month', 'Year', 'Hour']])

df['weekday'] = df['TimeStamp'].dt.dayofweek

#df.head()
# Eliminating national holidays



df_no_hol = df.drop(df[(df.Month == 1) & (df.Day == 1)].index)

df_no_hol = df_no_hol.drop(df_no_hol[(df_no_hol.Month == 5) & (df_no_hol.Day == 29)].index)

df_no_hol = df_no_hol.drop(df_no_hol[(df_no_hol.Month == 7) & (df_no_hol.Day == 4)].index)

df_no_hol = df_no_hol.drop(df_no_hol[(df_no_hol.Month == 9) & (df_no_hol.Day == 4)].index)

df_no_hol = df_no_hol.drop(df_no_hol[(df_no_hol.Month == 11) & (df_no_hol.Day == 23)].index)

df_no_hol = df_no_hol.drop(df_no_hol[(df_no_hol.Month == 12) & (df_no_hol.Day == 25)].index)



# Eliminating Weekends & National Holiday ( Days to be eliminated for peak hour calculation)



df_no_wend = df_no_hol.drop(df_no_hol[(df_no_hol.weekday == 5)].index)

df_no_wend = df_no_wend.drop(df_no_wend[(df_no_wend.weekday == 6)].index)



df_wd = df_no_wend # creating a copy of the dataframe
# selecting Peak Hours during Summer

df_SP = df_wd.drop(df_wd[((df_wd['Month'] < 5))].index)

df_SP = df_SP.drop(df_SP[((df_SP['Month'] > 10))].index)

df_SP = df_SP.drop(df_SP[((df_SP['Hour'] < 13))].index)

df_SP = df_SP.drop(df_SP[((df_SP['Hour'] > 19))].index)



# selecting Peak Hours during Winter

df_WP = df_wd.drop(df_wd[(df_wd['Month'] >= 5) & (df_wd['Month'] <= 10)].index)

df_WP = df_WP.drop(df_WP[(df_WP['Hour'] >= 9) & (df_WP['Hour'] < 17)].index)

df_WP = df_WP.drop(df_WP[(df_WP['Hour'] >= 21)].index)

df_WP = df_WP.drop(df_WP[(df_WP['Hour'] < 5)].index)



# Merging Peak Hour Data

df_Peak = pd.concat([df_SP, df_WP], axis=0, join='inner')





# Grouping by Month, to find the Maximum Hourly-Power-demand every month

df_month_max = pd.DataFrame(df_Peak.groupby(['Month'])['power_demand_kW'].max().reset_index())

# print (df_month_max)





# Calculating the Demand Charge

conditions = [

    (df_month_max['Month'] >= 5) & (df_month_max['Month'] < 7),

    (df_month_max['Month'] >= 9) & (df_month_max['Month'] < 11),   

    (df_month_max['Month'] >= 7) & (df_month_max['Month'] < 9)]



S_1 = df_month_max['power_demand_kW'] * 8.03

S_2 = df_month_max['power_demand_kW'] * 8.03

SP = df_month_max['power_demand_kW'] * 9.59

W = df_month_max['power_demand_kW'] * 3.55



choices = [S_1, S_2, SP]

df_month_max['Demand_Charge'] = np.select(conditions, choices, default = W)

df_month_max['Demand_Charge'] = df_month_max['Demand_Charge'] / 2      # reducing by half, for 30 mins window

print(df_month_max)
# visualize:



trace = go.Bar(

    x= df_month_max.Month,

    y= df_month_max.power_demand_kW,

#     yaxis = 'y1',

    marker = dict(color = 'orange',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.6,

    name = 'Max_Power(Hourly)'

    

)





trace2 = go.Scatter(

    x= df_month_max.Month,

    y= df_month_max.Demand_Charge,

    yaxis = 'y2',

    marker = dict(color = 'purple',

                 ),

    mode = 'lines+markers',

    name = 'Demand charge',

                 

)







layout = go.Layout(

    title = 'Monthly Maximum Power Demand (hourly)',

    yaxis = dict(title='Max_Hourly Power Demand (kWh)'),

    xaxis = dict(title='Month'),

    yaxis2= dict(title='Monthly Demand Charge (30 mins window) $', 

                 overlaying='y',

                 side ='right')

)



data = [trace, trace2]

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename="Max_Pwr_Dmnd(hourly)")
# breaking into different dataframes for segregating the data based on variety charge rates



# ------------------- National Holidays-------------------



hol_1 = df.loc[(df['Month']== 1) & (df['Day'] == 1)]      

hol_2 = df.loc[(df['Month']== 5) & (df['Day'] == 29)]

hol_3 = df.loc[(df['Month']== 7) & (df['Day'] == 4)]

hol_4 = df.loc[(df['Month']== 9) & (df['Day'] == 4)]

hol_5 = df.loc[(df['Month']== 11) & (df['Day'] == 23)]

hol_6 = df.loc[(df['Month']== 12) & (df['Day'] == 25)]



df_nat_hol = pd.concat([hol_1, hol_2, hol_3, hol_4, hol_5, hol_6], ignore_index=True)

df_nat_hol.head(n=50)





# ------------------- Excluding Weekends Only (Not Holidays)------------

df_no_wendT = df.drop(df[(df.weekday == 5)].index)

df_no_wendT = df_no_wendT.drop(df_no_wendT[(df_no_wendT.weekday == 6)].index)





# ------------------- Weekends Only ----------------------------

df_no_wdT = df.drop(df[(df.weekday == 0)].index)

df_no_wdT = df_no_wdT.drop(df_no_wdT[(df_no_wdT.weekday == 1)].index)

df_no_wdT = df_no_wdT.drop(df_no_wdT[(df_no_wdT.weekday == 2)].index)

df_no_wdT = df_no_wdT.drop(df_no_wdT[(df_no_wdT.weekday == 3)].index)

df_no_wdT = df_no_wdT.drop(df_no_wdT[(df_no_wdT.weekday == 4)].index)





# ------------- National Hols + Weekends (Dropping common days)--------------------

df_off_peak_hol = pd.concat([df_no_wdT, df_nat_hol],axis=0, join='inner')

df_off_peak_hol = df_off_peak_hol.drop_duplicates(keep='first')

df_off_peak_hol.describe()





# ------------------- Summer Peak------------------------

df_SPT = df_no_wend.drop(df_no_wend[( (df_no_wend['Month'] < 5))].index)

df_SPT = df_SPT.drop(df_SPT[((df_SPT['Month'] > 10))].index)

df_SPT = df_SPT.drop(df_SPT[((df_SPT['Month'] == 7))].index)

df_SPT = df_SPT.drop(df_SPT[((df_SPT['Month'] == 8))].index)

df_SPT = df_SPT.drop(df_SPT[((df_SPT['Hour'] < 13))].index)

df_SPT = df_SPT.drop(df_SPT[((df_SPT['Hour'] > 19))].index)

df_SPT['Unit_Charge'] = 0.0486





# ------------------- Summer Off Peak------------------------

df_off_SPT_wd = df_no_wend.drop(df_no_wend[((df_no_wend['Month'] < 5))].index)

df_off_SPT_wd = df_off_SPT_wd.drop(df_off_SPT_wd[((df_off_SPT_wd['Month'] > 10))].index)

df_off_SPT_wd = df_off_SPT_wd.drop(df_off_SPT_wd[((df_off_SPT_wd['Month'] == 7))].index)

df_off_SPT_wd = df_off_SPT_wd.drop(df_off_SPT_wd[((df_off_SPT_wd['Month'] == 8))].index)

df_off_SPT_wd = df_off_SPT_wd.drop(df_off_SPT_wd[((df_off_SPT_wd['Hour'] >= 13) & (df_off_SPT_wd['Hour'] < 20))].index)

df_off_SPT_wd['Unit_Charge'] = 0.0371





df_off_SPT_hol = df_off_peak_hol.drop(df_off_peak_hol[((df_off_peak_hol['Month'] < 5))].index)

df_off_SPT_hol = df_off_SPT_hol.drop(df_off_SPT_hol[((df_off_SPT_hol['Month'] > 10))].index)

df_off_SPT_hol = df_off_SPT_hol.drop(df_off_SPT_hol[((df_off_SPT_hol['Month'] == 7))].index)

df_off_SPT_hol = df_off_SPT_hol.drop(df_off_SPT_hol[((df_off_SPT_hol['Month'] == 8))].index)

df_off_SPT_hol['Unit_Charge'] = 0.0371





# ------------------- Peak Summer Peak------------------------

df_PSPT = df_no_wend.drop(df_no_wend[( (df_no_wend['Month'] < 7))].index)

df_PSPT = df_PSPT.drop(df_PSPT[((df_PSPT['Month'] > 8))].index)

df_PSPT = df_PSPT.drop(df_PSPT[((df_PSPT['Hour'] < 13))].index)

df_PSPT = df_PSPT.drop(df_PSPT[((df_PSPT['Hour'] > 19))].index)

df_PSPT['Unit_Charge'] = 0.0633





# ------------------- Peak Summer Off Peak------------------------

df_off_PSPT_wd = df_no_wend.drop(df_no_wend[((df_no_wend['Month'] < 7))].index)

df_off_PSPT_wd = df_off_PSPT_wd.drop(df_off_PSPT_wd[((df_off_PSPT_wd['Month'] > 8))].index)

df_off_PSPT_wd = df_off_PSPT_wd.drop(df_off_PSPT_wd[((df_off_PSPT_wd['Hour'] >= 13) & (df_off_PSPT_wd['Hour'] < 20))].index)

df_off_PSPT_wd['Unit_Charge'] = 0.0423





df_off_PSPT_hol = df_off_peak_hol.drop(df_off_peak_hol[((df_off_peak_hol['Month'] < 7))].index)

df_off_PSPT_hol = df_off_PSPT_hol.drop(df_off_PSPT_hol[((df_off_PSPT_hol['Month'] > 8))].index)

df_off_PSPT_hol['Unit_Charge'] = 0.0423







#------------------- Winter Peak -----------------------

df_WPT = df_no_wend.drop(df_no_wend[(df_no_wend['Month'] >= 5) & (df_no_wend['Month'] <= 10)].index)

df_WPT = df_WPT.drop(df_WPT[(df_WPT['Hour'] >= 9) & (df_WPT['Hour'] < 17)].index)

df_WPT = df_WPT.drop(df_WPT[(df_WPT['Hour'] >= 21)].index)

df_WPT = df_WPT.drop(df_WPT[(df_WPT['Hour'] < 5)].index)

df_WPT['Unit_Charge'] = 0.0410





# ------------------- Winter Off Peak --------------------------------------

df_off_WPT_wd = df_no_wend.drop(df_no_wend[(df_no_wend['Month'] >= 5) & (df_no_wend['Month'] <= 10)].index)

df_off_WPT_wd = df_off_WPT_wd.drop(df_off_WPT_wd[(df_off_WPT_wd['Hour'] >= 5) & (df_off_WPT_wd['Hour'] < 9)].index)

df_off_WPT_wd = df_off_WPT_wd.drop(df_off_WPT_wd[(df_off_WPT_wd['Hour'] >= 17)& (df_off_WPT_wd['Hour'] < 21)].index)

df_off_WPT_wd['Unit_Charge'] = 0.0370





df_off_WPT_hol = df_off_peak_hol.drop(df_off_peak_hol[(df_off_peak_hol['Month'] >= 5) & (df_off_peak_hol['Month'] <= 10)].index)

df_off_WPT_hol['Unit_Charge'] = 0.0370
df_all = pd.concat([df_SPT, df_off_SPT_wd, df_off_SPT_hol,

                            df_PSPT, df_off_PSPT_wd, df_off_PSPT_hol, 

                            df_WPT, df_off_WPT_wd, df_off_WPT_hol ], axis=0, join='inner')



df_all['Consumption_Charge'] = df_all['power_demand_kW'] * df_all['Unit_Charge']



df_month_Consumption = pd.DataFrame(df_all.groupby(['Month'])['Consumption_Charge'].sum().reset_index())

print (df_month_Consumption)
trace3 = go.Bar(

    x= df_month_Consumption.Month,

    y= df_month_Consumption.Consumption_Charge,

#     yaxis = 'y1',

    marker = dict(color = 'green',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.6,

    name = 'Total_Power_Demand'

    

)





trace4 = go.Scatter(

    x= df_month_max.Month,

    y= df_month_max.Demand_Charge,

    yaxis = 'y2',

    marker = dict(color = 'brown',

                 ),

    mode = 'lines+markers',

    name = 'Total Consumption Charge',

                 

)







layout = go.Layout(

    title = 'Monthly Power Consumption',

    yaxis = dict(title='Total Power Demand (kWh)'),

    xaxis = dict(title='Month'),

    yaxis2= dict(title='Monthly Power Charge $', 

                 overlaying='y',

                 side ='right')

)



data = [trace3, trace4]

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename="Max_Pwr_Dmnd(hourly)")
df_monthly_bill = df_month_max.merge(df_month_Consumption, how='left')

df_monthly_bill = df_monthly_bill.drop(columns=['power_demand_kW'])

df_monthly_bill['Meter_Charge'] = 32.44

df_monthly_bill['Total_Bill'] = df_monthly_bill['Meter_Charge'] + df_monthly_bill['Demand_Charge'] + df_monthly_bill['Consumption_Charge']

print (df_monthly_bill)



trace9 = go.Bar(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Demand_Charge,

    marker = dict(color = 'pink',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.9,

    name = 'Demand_Charge'

    

)



trace10 = go.Bar(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Consumption_Charge,

    marker = dict(color = 'red',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.8,

    name = 'Consumption Charge'

    

)





trace11 = go.Bar(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Meter_Charge,

    marker = dict(color = 'purple',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.6,

    name = 'Meter Charge'

    

)





trace12 = go.Scatter(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Total_Bill,

    marker = dict(color = 'brown',

                 ),

    mode = 'lines+markers',

    name = 'Total Monthly Bill',

                 

)







layout = go.Layout(

    title = 'Monthly Billing Charges Breakdown (2017)',

    yaxis = dict(title='Charges ($)'),

    xaxis = dict(title='Month'),

)



data = [trace9, trace10, trace11, trace12]

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename="Billing Charge Division (2017)")
# Calling the raw dataset

df_all = df_all.sort_values(['TimeStamp'])

df_all.head()
df_2 = df_all.rename(index=str, columns={"power_demand_kW": "User_Demand_L", "solar_kW": "Produced_Energy_G"})

df_2 = df_2.drop(['Year', 'Consumption_Charge'], axis=1)



df_2['Excess_Production_E'] = df_2['Produced_Energy_G'] - df_2['User_Demand_L']  # Creating a column with excess power Generation

df_2['Excess_Production_E'] = df_2['Excess_Production_E'].clip_lower(0) # Eliminating negative values

df_2['battery_storage'] = 3.3  # Benchmarking the minimum storage energy required for battery to run

df_2['Billing_Demand'] = float()
# Linear Model for billing after solar installation

for t in range (0, len(df_2)-1):

    if df_2['Excess_Production_E'][t+1]>0: 

        if df_2['battery_storage'][t] < 9.3 :

            df_2['battery_storage'][t+1]= df_2['battery_storage'][t] + df_2['Excess_Production_E'][t+1]

            df_2['Billing_Demand'][t+1] = 0

            if df_2['battery_storage'][t+1] > 9.3:

                df_2['battery_storage'][t+1]  = 9.3

            else :

                continue

              

        elif df_2['battery_storage'][t] == 9.3 :

            df_2['battery_storage'][t+1] = 9.3

            df_2['Billing_Demand'][t+1] = 0

        

    else:

        if (df_2['battery_storage'][t] - 3.3) > df_2['User_Demand_L'][t+1] :

            df_2['Billing_Demand'][t+1] = 0

            df_2['battery_storage'][t+1] = df_2['battery_storage'][t]-df_2['User_Demand_L'][t+1]

        

        elif (df_2['battery_storage'][t] - 3.3) < df_2['User_Demand_L'][t+1] :

            df_2['battery_storage'][t+1] = 3.3

            df_2['Billing_Demand'][t+1] = df_2['User_Demand_L'][t+1] - (df_2['battery_storage'][t] - 3.3)

df_2['Adjusted_Consumption_Charge'] = df_2['Billing_Demand'] * df_2['Unit_Charge']

df_month_Billing = pd.DataFrame(df_2.groupby(['Month'])['Adjusted_Consumption_Charge'].sum().reset_index())



print (df_month_Billing)
trace5 = go.Bar(

    x= df_month_Consumption.Month,

    y= df_month_Consumption.Consumption_Charge,

    marker = dict(color = 'red',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.6,

    name = 'Before_Solar_installation',

    

)





trace6 = go.Bar(

    x= df_month_Billing.Month,

    y= df_month_Billing.Adjusted_Consumption_Charge,

    marker = dict(color = 'yellow',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity = 0.8,

    name = 'After_Solar_installation',

                 

)



layout = go.Layout(

    title = 'Comparison of Monthly Power Consumption Charges',

    yaxis = dict(title='Monthly_Consumption_Amount ($)'),

    xaxis = dict(title='Month')

)



data = [trace5, trace6]

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename="Compare_Consumption_Plot")

    



# Eliminating national holidays



df_no_hol_2 = df_2.drop(df_2[(df_2.Month == 1) & (df_2.Day == 1)].index)

df_no_hol_2 = df_no_hol_2.drop(df_no_hol_2[(df_no_hol_2.Month == 5) & (df_no_hol_2.Day == 29)].index)

df_no_hol_2 = df_no_hol_2.drop(df_no_hol_2[(df_no_hol_2.Month == 7) & (df_no_hol_2.Day == 4)].index)

df_no_hol_2 = df_no_hol_2.drop(df_no_hol_2[(df_no_hol_2.Month == 9) & (df_no_hol_2.Day == 4)].index)

df_no_hol_2 = df_no_hol_2.drop(df_no_hol_2[(df_no_hol_2.Month == 11) & (df_no_hol_2.Day == 23)].index)

df_no_hol_2 = df_no_hol_2.drop(df_no_hol_2[(df_no_hol_2.Month == 12) & (df_no_hol_2.Day == 25)].index)



# Eliminating Weekends & National Holiday



df_no_wend_2 = df_no_hol_2.drop(df_no_hol_2[(df_no_hol_2.weekday == 5)].index)

df_no_wend_2 = df_no_wend_2.drop(df_no_wend_2[(df_no_wend_2.weekday == 6)].index)



df_wd_2 = df_no_wend_2



# selecting Peak Hours during Summer

df_SP_2 = df_wd_2.drop(df_wd_2[( (df_wd_2['Month'] < 5))].index)

df_SP_2 = df_SP_2.drop(df_SP_2[((df_SP_2['Month'] > 10))].index)

df_SP_2 = df_SP_2.drop(df_SP_2[((df_SP_2['Hour'] < 13))].index)

df_SP_2 = df_SP_2.drop(df_SP_2[((df_SP_2['Hour'] > 19))].index)



# selecting Peak Hours during Winter

df_WP_2 = df_wd_2.drop(df_wd_2[(df_wd_2['Month'] >= 5) & (df_wd_2['Month'] <= 10)].index)

df_WP_2 = df_WP_2.drop(df_WP_2[(df_WP_2['Hour'] >= 9) & (df_WP_2['Hour'] < 17)].index)

df_WP_2 = df_WP_2.drop(df_WP_2[(df_WP_2['Hour'] >= 21)].index)

df_WP_2 = df_WP_2.drop(df_WP_2[(df_WP_2['Hour'] < 5)].index)



# Merging Peak Hour Data

df_Peak_2 = pd.concat([df_SP_2, df_WP_2], axis=0, join='inner')



df_adjusted_month_max = pd.DataFrame(df_Peak_2.groupby(['Month'])['Billing_Demand'].max().reset_index())



conditions = [

    (df_adjusted_month_max['Month'] >= 5) & (df_adjusted_month_max['Month'] < 7),

    (df_adjusted_month_max['Month'] >= 9) & (df_adjusted_month_max['Month'] < 11),   

    (df_adjusted_month_max['Month'] >= 7) & (df_adjusted_month_max['Month'] < 9)]



S_1 = df_adjusted_month_max['Billing_Demand'] * 8.03

S_2 = df_adjusted_month_max['Billing_Demand'] * 8.03

SP = df_adjusted_month_max['Billing_Demand'] * 9.59

W = df_adjusted_month_max['Billing_Demand'] * 3.55



choices = [S_1, S_2, SP]

df_adjusted_month_max['Adjusted_Demand_Charge'] = np.select(conditions, choices, default = W)

df_adjusted_month_max['Adjusted_Demand_Charge'] = df_adjusted_month_max['Adjusted_Demand_Charge'] / 2      # reducing by half, for 30 mins window

print(df_adjusted_month_max)
trace7 = go.Scatter(

    x= df_month_max.Month,

    y= df_month_max.Demand_Charge,

    marker = dict(color = 'red',

                 ),

    mode = 'lines+markers',

    name = 'Before_Solar_installation',

    

)





trace8 = go.Scatter(

    x= df_adjusted_month_max.Month,

    y= df_adjusted_month_max.Adjusted_Demand_Charge,

    marker = dict(color = 'green',

                 ),

    mode = 'lines+markers',

    name = 'After_Solar_installation',

          

)



layout = go.Layout(

    title = 'Comparison of Monthly Demand Charges',

    yaxis = dict(title='Monthly_Demand_Charge ($)'),

    xaxis = dict(title='Month')

)



data = [trace7, trace8]

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename="Compare_Consumption_Plot")

    
df_adjusted_monthly_bill = df_adjusted_month_max.merge(df_month_Billing, how='left')

df_adjusted_monthly_bill = df_adjusted_monthly_bill.drop(columns=['Billing_Demand'])

df_adjusted_monthly_bill['Meter_Charge'] = 32.44

df_adjusted_monthly_bill['Adjusted_Total_Bill'] = df_adjusted_monthly_bill['Meter_Charge'] + df_adjusted_monthly_bill['Adjusted_Demand_Charge'] + df_adjusted_monthly_bill['Adjusted_Consumption_Charge']

print (df_adjusted_monthly_bill)
trace13 = go.Bar(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Demand_Charge,

    marker = dict(color = 'pink',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.9,

    name = 'Demand_Charge'

    

)



trace14 = go.Bar(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Consumption_Charge,

    marker = dict(color = 'purple',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.8,

    name = 'Consumption Charge'

    

)







trace15 = go.Bar(

    x= df_adjusted_monthly_bill.Month,

    y= df_adjusted_monthly_bill.Adjusted_Demand_Charge,

    marker = dict(color = 'yellow',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.4,

    name = 'Adjusted_Demand_Charge'

    

)



trace16 = go.Bar(

    x= df_adjusted_monthly_bill.Month,

    y=df_adjusted_monthly_bill.Adjusted_Consumption_Charge,

    marker = dict(color = 'orange',

                  line = dict(

                      color ='rgb(8,4,107)',

                      width=1.5),

                 ),

    opacity =0.8,

    name = 'Adjusted_Consumption Charge'

    

)







trace17 = go.Scatter(

    x= df_monthly_bill.Month,

    y= df_monthly_bill.Total_Bill,

    marker = dict(color = 'brown',

                 ),

    mode = 'lines+markers',

    name = 'Total Monthly Bill',

                 

)



trace18 = go.Scatter(

    x= df_adjusted_monthly_bill.Month,

    y= df_adjusted_monthly_bill.Adjusted_Total_Bill ,

    marker = dict(color = 'green',

                 ),

    mode = 'lines+markers',

    name = 'Total Adjusted Monthly Bill',

                 

)





layout = go.Layout(

    title = 'Comparison of Total Monthly Bills and its breakdowns',

    yaxis = dict(title='Charges ($)'),

    xaxis = dict(title='Month'),

)



data = [trace13, trace14, trace15, trace16, trace17, trace18]

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename="Adjusted Billing Charge Division (2017)")
Yearly_Cost_Before_Solar= df_monthly_bill['Total_Bill'].sum()

Yearly_Cost_After_Solar = df_adjusted_monthly_bill['Adjusted_Total_Bill'].sum()



Yearly_Saving = Yearly_Cost_Before_Solar - Yearly_Cost_After_Solar

print(Yearly_Saving)
objects = ('After_Solar_System','Before_Solar_System')

y_pos = np.arange(len(objects))

performance = [Yearly_Cost_After_Solar, Yearly_Cost_Before_Solar]

 

plt.bar(y_pos, performance, align='center', alpha=0.9)

plt.xticks(y_pos, objects)

plt.ylabel('Yearly_total ($)')

plt.title('Billing Comparison')

 

plt.show()