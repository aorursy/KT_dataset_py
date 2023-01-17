import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

from IPython.display import Markdown, display

from scipy.optimize import curve_fit



pd.options.mode.chained_assignment = None



# Read the data

us_data_path = "/kaggle/input/covid19-in-usa/"

us_df = pd.read_csv(us_data_path + "us_covid19_daily.csv")

us_states_df = pd.read_csv(us_data_path + "us_states_covid19_daily.csv")

us_df["date"] = pd.to_datetime(us_df["date"], format="%Y%m%d")

us_states_df = us_states_df.reindex(index=us_states_df.index[::-1])

us_states_df["date"] = pd.to_datetime(us_states_df["date"], format="%Y%m%d").dt.date.astype(str)

#us_states_df.head()



# US state code to name mapping

state_map_dict = {'AL': 'Alabama',

 'AK': 'Alaska',

 'AS': 'American Samoa',

 'AZ': 'Arizona',

 'AR': 'Arkansas',

 'CA': 'California',

 'CO': 'Colorado',

 'CT': 'Connecticut',

 'DE': 'Delaware',

 'DC': 'District of Columbia',

 'D.C.': 'District of Columbia',

 'FM': 'Federated States of Micronesia',

 'FL': 'Florida',

 'GA': 'Georgia',

 'GU': 'Guam',

 'HI': 'Hawaii',

 'ID': 'Idaho',

 'IL': 'Illinois',

 'IN': 'Indiana',

 'IA': 'Iowa',

 'KS': 'Kansas',

 'KY': 'Kentucky',

 'LA': 'Louisiana',

 'ME': 'Maine',

 'MH': 'Marshall Islands',

 'MD': 'Maryland',

 'MA': 'Massachusetts',

 'MI': 'Michigan',

 'MN': 'Minnesota',

 'MS': 'Mississippi',

 'MO': 'Missouri',

 'MT': 'Montana',

 'NE': 'Nebraska',

 'NV': 'Nevada',

 'NH': 'New Hampshire',

 'NJ': 'New Jersey',

 'NM': 'New Mexico',

 'NY': 'New York',

 'NC': 'North Carolina',

 'ND': 'North Dakota',

 'MP': 'Northern Mariana Islands',

 'OH': 'Ohio',

 'OK': 'Oklahoma',

 'OR': 'Oregon',

 'PW': 'Palau',

 'PA': 'Pennsylvania',

 'PR': 'Puerto Rico',

 'RI': 'Rhode Island',

 'SC': 'South Carolina',

 'SD': 'South Dakota',

 'TN': 'Tennessee',

 'TX': 'Texas',

 'UT': 'Utah',

 'VT': 'Vermont',

 'VI': 'Virgin Islands',

 'VA': 'Virginia',

 'WA': 'Washington',

 'WV': 'West Virginia',

 'WI': 'Wisconsin',

 'WY': 'Wyoming'}



state_code_dict = {v:k for k, v in state_map_dict.items()}

state_code_dict["Chicago"] = 'Illinois'



def correct_state_names(x):

    try:

        return state_map_dict[x.split(",")[-1].strip()]

    except:

        return x.strip()

    

# us_covid19_daily.csv has numbers for states instead of names

def get_state_codes(x):

    try:

        return state_code_dict[x]

    except:

        return "Others"

    

#---print stats and such in markdown for daily updates that are easier to see    

def printmd(string):

    display(Markdown(string))

    

# ---------- color print large text -----------------

def printmdc(string, color=None):

    colorstr = "## <span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))



# ----for curve fitting-------    

def func(x, a, b, c, d):

#     return a * np.exp(b * x) + c

    return a*b**(c*x) + d



# read the global data set

covid_19_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

# Extract only the US cases

us_covid_df = covid_19_df[covid_19_df["Country/Region"]=="US"]

# Correct state names and apply state code

us_covid_df["Province/State"] = us_covid_df["Province/State"].apply(correct_state_names)

us_covid_df["StateCode"] = us_covid_df["Province/State"].apply(lambda x: get_state_codes(x))



cumulative_df = us_covid_df.groupby("ObservationDate")["Confirmed", "Deaths", "Recovered"].sum().reset_index()



#fit the cumulative covid cases data to an exponential

xdata = cumulative_df["Confirmed"].index.values 

ydata = cumulative_df["Confirmed"].values

popt, pcov = curve_fit(func, xdata, ydata, p0=(0, 2, 0,0)) 

# stdv_error = np.sqrt(np.diag(pcov))

# print("std of fit error: " + str(stdv_error))

print("Covariance Matrix (squared errors are diagnols): " + str(popt))



### Plot for number of cumulative covid cases over time

fig = px.bar(cumulative_df, x="ObservationDate", y="Confirmed",color = "Deaths")

fig.add_trace(px.line(x=cumulative_df.ObservationDate, y=func(xdata, *popt)).data[0]) #comment out this line to remove fit

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in US",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)

fig.update_layout(layout)

fig.show()



### Plot for number of cumulative deaths over time

# fig = px.bar(cumulative_df, x="ObservationDate", y="Deaths")

# layout = go.Layout(

#     title=go.layout.Title(

#         text="Daily cumulative count of deaths due to COVID-19 in US",

#         x=0.5

#     ),

#     font=dict(size=14),

#     width=750,

#     height=450,

#     xaxis_title = "Date of observation",

#     yaxis_title = "Number of death cases"

# )

# fig.update_layout(layout)

# fig.show()



### Plot for number of confirmed new cases over time

cumulative_df["ConfirmedNew"] = cumulative_df["Confirmed"].diff() 

fig = px.bar(cumulative_df, x="ObservationDate", y="ConfirmedNew")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily count of new confirmed COVID-19 cases in US",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()

print("Exponential fit parameters (a,b,c,d) for a*b^(c*x) + d: " + str(popt))

# printmd("## Number of confirmed cases in US: " + str(int(cumulative_df.Confirmed.iloc[-1])))

print("Number of conrimed cases in US: " + str(str(int(cumulative_df.Confirmed.iloc[-1]))))

tomorrows_day = len(cumulative_df.index)

pred_tmrw = func(tomorrows_day, *popt)

print("(what the fit predicts for today: " + str(func(tomorrows_day-1, *popt)) +  ")")

# printmdc("Number of deaths in US: " + str(int(cumulative_df.Deaths.iloc[-1])),color='red')

print("Number of deaths in US: " + str(int(cumulative_df.Deaths.iloc[-1])))

tomorrows_day = len(cumulative_df.index)

pred_tmrw = func(tomorrows_day, *popt)

# printmdc("Predicted confirmed cases for tomorrow: " + str(int(pred_tmrw)))

print("Predicted confirmed cases for tomorrow: " + str(int(pred_tmrw)))
# # Read the data

# # source 1: covid19-in-usa

# us_data_path = "/kaggle/input/covid19-in-usa/"

# us_df = pd.read_csv(us_data_path + "us_covid19_daily.csv")

# us_df = us_df.drop('posNeg',axis=1)

# us_df["StateCode"] = us_df["Province/State"].apply(lambda x: get_state_codes(x))

# print(us_df.head())



# # source 2: us_states_covid19_daily

# us_states_df = pd.read_csv(us_data_path + "us_states_covid19_daily.csv")

# print(us_states_df.head())



# # source 3: global source: 

# us_df["date"] = pd.to_datetime(us_df["date"], format="%Y%m%d")

# us_states_df = us_states_df.reindex(index=us_states_df.index[::-1])

# us_states_df["date"] = pd.to_datetime(us_states_df["date"], format="%Y%m%d").dt.date.astype(str)

# # print(us_states_df.head())



# # check to see overlaps

# us_df.equals(us_states_df)
ny_covid_df = us_covid_df.loc[us_covid_df.StateCode == 'NY']

ny_cum_df = ny_covid_df.groupby("ObservationDate")["Confirmed", "Deaths", "Recovered"].sum().reset_index()



xdata = ny_cum_df["Confirmed"].index.values 

ydata = ny_cum_df["Confirmed"].values

popt, pcov = curve_fit(func, xdata, ydata) 

#---------plot without fit

### Plot for number of cumulative covid cases over time

# fig = px.bar(ny_cum_df, x="ObservationDate", y="Confirmed")

# layout = go.Layout(

#     title=go.layout.Title(

#         text="Daily cumulative count of confirmed COVID-19 cases in NY",

#         x=0.5

#     ),

#     font=dict(size=14),

#     width=800,

#     height=500,

#     xaxis_title = "Date of observation",

#     yaxis_title = "Number of confirmed cases"

# )

# fig.update_layout(layout)

# fig.show()



#---------plot with fit-------------------------------

### Plot for number of cumulative covid cases over time,

fig = px.bar(ny_cum_df, x="ObservationDate", y="Confirmed",color = "Deaths")

fig.add_trace(px.line(x=ny_cum_df.ObservationDate, y=func(xdata, *popt)).data[0])

layout = go.Layout(

        title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in NY",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)

fig.update_layout(layout)

fig.show()







### Cumulative deaths

# fig = px.bar(ny_cum_df, x="ObservationDate", y="Deaths")

# layout = go.Layout(

#     title=go.layout.Title(

#         text="Daily cumulative count of deaths due to COVID-19 in NY",

#         x=0.5

#     ),

#     font=dict(size=14),

#     width=750,

#     height=450,

#     xaxis_title = "Date of observation",

#     yaxis_title = "Number of death cases"

# )

# fig.update_layout(layout)

# fig.show()



### Plot for number of cumulative covid cases over time

ny_cum_df["ConfirmedNew"] = ny_cum_df["Confirmed"].diff() 

fig = px.bar(ny_cum_df, x="ObservationDate", y="ConfirmedNew")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily count of new confirmed COVID-19 cases in NY",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()



print("Exponential fit parameters (a,b,c,d) for a*b^(c*x) + d: " + str(popt))

# printmd("## Number of confirmed cases in NY: " + str(int(ny_cum_df.Confirmed.iloc[-1])))

print("Number of confirmed cases in NY: " + str(int(ny_cum_df.Confirmed.iloc[-1])))

# printmdc("Number of deaths in NY: " + str(int(ny_cum_df.Deaths.iloc[-1])),color='red')

print("Number of deaths in NY: " + str(int(ny_cum_df.Deaths.iloc[-1])))

tomorrows_day = len(ny_cum_df.index)

pred_tmrw = func(tomorrows_day, *popt)

# printmdc("Predicted confirmed cases for tomorrow: " + str(int(pred_tmrw)))

print("Predicted confirmed cases for tomorrow: " + str(int(pred_tmrw)))

print("(what the fit predicts for today: " + str(func(tomorrows_day-1, *popt)) +  ")")
ma_covid_df = us_covid_df.loc[us_covid_df.StateCode == 'MA']

ma_cum_df = ma_covid_df.groupby("ObservationDate")["Confirmed", "Deaths", "Recovered"].sum().reset_index()



### Plot for number of cumulative covid cases over time

fig = px.bar(ma_cum_df, x="ObservationDate", y="Confirmed")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in MA",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()



### Plot for number of cumulative covid cases over time

fig = px.bar(ma_cum_df, x="ObservationDate", y="Deaths")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of deaths due to COVID-19 in MA",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of death cases"

)



fig.update_layout(layout)

fig.show()



### Plot for number of cumulative covid cases over time

ma_cum_df["ConfirmedNew"] = ma_cum_df["Confirmed"].diff() 

fig = px.bar(ma_cum_df, x="ObservationDate", y="ConfirmedNew")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily count of new confirmed COVID-19 cases in MA",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()



    

# printmd("Number of confirmed cases in MA: " + str(ma_cum_df.Confirmed.iloc[-1]), color="Red")

# printmd("## Number of confirmed cases in MA: " + str(int(ma_cum_df.Confirmed.iloc[-1])))

print("Number of confirmed cases in MA: " + str(int(ma_cum_df.Confirmed.iloc[-1])))

# printmd("## Number of deaths in MA: " + str(int(ma_cum_df.Deaths.iloc[-1])))

print("Number of deaths in MA: " + str(int(ma_cum_df.Deaths.iloc[-1])))
nj_covid_df = us_covid_df.loc[us_covid_df.StateCode == 'NJ']

nj_cum_df = nj_covid_df.groupby("ObservationDate")["Confirmed", "Deaths", "Recovered"].sum().reset_index()



xdata = nj_cum_df["Confirmed"].index.values 

ydata = nj_cum_df["Confirmed"].values

popt, pcov = curve_fit(func, xdata, ydata) 





#---------plot with fit-------------------------------

### Plot for number of cumulative covid cases over time,

fig = px.bar(nj_cum_df, x="ObservationDate", y="Confirmed",color = "Deaths")

fig.add_trace(px.line(x=nj_cum_df.ObservationDate, y=func(xdata, *popt)).data[0])

layout = go.Layout(

        title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in NJ",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)

fig.update_layout(layout)

fig.show()





### Plot for number of cumulative covid cases over time

nj_cum_df["ConfirmedNew"] = nj_cum_df["Confirmed"].diff() 

fig = px.bar(nj_cum_df, x="ObservationDate", y="ConfirmedNew")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily count of new confirmed COVID-19 cases in NJ",

        x=0.5

    ),

    font=dict(size=14),

    width=750,

    height=450,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()



# printmd("## Number of confirmed cases in NJ: " + str(int(nj_cum_df.Confirmed.iloc[-1])))

print("Number of confirmed cases in NJ: " + str(int(nj_cum_df.Confirmed.iloc[-1])))

# printmdc("Number of deaths in NJ: " + str(int(nj_cum_df.Deaths.iloc[-1])),color='red')

print("Number of deaths in NJ: " + str(int(nj_cum_df.Deaths.iloc[-1])))

tomorrows_day = len(nj_cum_df.index)

pred_tmrw = func(tomorrows_day, *popt)

# printmdc("Predicted confirmed cases for tomorrow: " + str(int(pred_tmrw)))

print("Predicted confirmed cases for tomorrow: " + str(int(pred_tmrw)))

print("(what the fit predicts for today: " + str(func(tomorrows_day-1, *popt)) +  ")")
#define a function to fit. In this case an exponential.

def func(x, a, b, c):

    return a * np.exp(b * x) + c



# instead of taking the dates as the xdata, it's easier to just take

# the index values since the x-axis is day-by-day

xdata = ny_cum_df["Confirmed"].index.values 

ydata = ny_cum_df["Confirmed"].values

#popt = optimal vals for fitted params

#pcov = The estimated covariance of popt. 

#The diagonals provide the variance of the parameter estimate.

# To compute one standard deviation errors on the parameters use 

#perr = np.sqrt(np.diag(pcov))

popt, pcov = curve_fit(func, xdata, ydata) 

stdv_error = np.sqrt(np.diag(pcov))

print("std of error: " + str(stdv_error))

print("Exponential fit parameters (a,b,c): " + str(popt))



#---------------------------Seaborn--------------------------

sns.set()

plt.figure(figsize=(12,6))

plt.title('Confirmed Cases in NY')

sns.set_style('darkgrid')

# sns.set_context("talk",font_scale = 4)

barplot_ny = sns.barplot(ny_cum_df.ObservationDate, ydata)

plt.plot(xdata, func(xdata, *popt), 'r-')

barplot_ny.set_xticklabels(barplot_ny.get_xticklabels(), rotation=270)



#----------------------------Plotly--------------------------

### Plot for number of cumulative covid cases over time,

fig = px.bar(ny_cum_df, x="ObservationDate", y="Confirmed",color = "Deaths") #you can remove the color param to simplify things

fig.add_trace(px.line(x=ny_cum_df.ObservationDate, y=func(xdata, *popt)).data[0])

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in NY",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)

fig.update_layout(layout)

fig.show()







tomorrows_day = len(ny_cum_df.index)+1

pred_tmrw = func(tomorrows_day, *popt)


