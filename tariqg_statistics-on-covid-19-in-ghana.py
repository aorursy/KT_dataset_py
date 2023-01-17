import requests
import bs4
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
from math import floor

init_notebook_mode(connected=True)
# Regular expression used to only select numeric data in a list
def getNumbers(array): 
    arr = re.findall(r'[0-9]', array) 
    return arr 

# Removes commas in lists
def clean(data):
    strings = [str(num) for num in data]
    concatenated = "".join(strings)
    return(concatenated)
res = requests.get('https://www.worldometers.info/coronavirus/country/ghana/')
text = bs4.BeautifulSoup(res.text)
cases = clean(getNumbers(str(text.select('div span')[4])))
deaths = clean(getNumbers(str(text.select('div span')[5])))
recoveries = clean(getNumbers(str(text.select('div span')[6])))

print("This is Ghana's current Covid-19 situation:\n")
print("Total Cases:", cases, "\n")
print("Total Recoveries:", recoveries, "\n")
print("Total Deaths:", deaths,'\n')
js = text.select('script[type="text/javascript"]') #Js contains the javascript of the website

#Js is a list and the various list indices extract the useful part of the list needed
data_for_days = list(js[7])
data_for_cases = list(js[7])
data_for_deaths = list(js[10])
active = list(js[9])

#Regular expression \w\w\w\s\d+ matches any three digit word with a space followed by one or more numbers
days_reg = re.compile(r'\w\w\w\s\d+')
days = days_reg.findall(data_for_days[0])

#Regular expression d{1,7} matches any number with at most 7 digits. 
cases_reg = re.findall(r'\d{1,7}',data_for_cases[0]) 
deaths_reg = re.findall(r'\d{1,7}',data_for_deaths[0]) 
active_reg = re.findall(r'\d{1,7}',active[0])

data_for_cases = [int(i) for i in cases_reg]
data_for_deaths = [int(i) for i in deaths_reg]

#unwanted data from the 0th to the 350th elements are removed
data_for_cases = data_for_cases[350::]
data_for_deaths = data_for_deaths[350::]

data_for_active = [int(i) for i in active_reg]

#Unwanted data from the 0th to the 222th elements are removed
data_for_active = data_for_active[223::]
active_index = data_for_active.index(2)

case_index = data_for_cases.index(0)
death_index = data_for_deaths.index(0)
total_cases = data_for_cases[case_index:len(data_for_cases) - 1]
total_cases = [int(i) for i in total_cases]

total_deaths = data_for_deaths[death_index:len(data_for_deaths) - 1]
total_deaths = [int(i) for i in total_deaths]

active_cases = data_for_active[active_index:len(data_for_active) - 1]
active_cases = [int(i) for i in active_cases]

#The required data starts from the 26th element
total_cases = total_cases[26::]
total_deaths = total_deaths[26::]

#The first case of covid-19 was recorded in Ghana on March 12,2020
ind = days.index('Jul 29')
days = days[ind::]

ind = days.index('Mar 12')
days = days[ind::]
data = {'Dates':days,
        'Total Cases':total_cases,
        'Total Deaths': total_deaths,
         'Active Cases': active_cases}

df = pd.DataFrame(data)
# Finds the number of daily cases based on the difference of a current number
# and it's previous number 
def daily_values(data,array):
    for i in range(len(data) - 1):
        hold = data[i+1] - data[i]
        array.append(hold)
daily_cases = []
daily = list(df["Total Cases"])

daily_values(daily,daily_cases) 
# 2 is inserted at the first position because Ghana first recorded 2 cases
daily_cases.insert(0,2)

df["Daily Cases"] = daily_cases
'''
Uses recursion to find the cumulative sum of all the elements in an array from 
their respective indices to the start of the array. This is used in the 
calculation of average daily statistics as everyday's value can be summed with
all the previous values and divided by the index + 1.

Example:
Input: [2,14,17,36]

Output: [2, 14+2, 17+14+2, 36+17+14+2]

nb: division of the values is not performed in the cumulative_sum function
    but in the daily_average function

'''

def cumulative_sum(data,array):
    total = 0
    values = []
    for i,j in enumerate(data):
        total = total+j 
        values.append(total)
    if total <= 1000 | total<=100:
        array.append(values)
        return cumulative_sum(data,array)
    else:
        return array.append(values)
    
'''
Finds the average of every element in an array. Each element
is divided by the index + 1

'''
def daily_average(data,array):
    for i in data:
        for y,k in enumerate(i):
            array.append( (k) // (y+1) )    
            
# average daily cases
daily_cases_sum = []
cumulative_sum(daily_cases,daily_cases_sum)

avg_daily_cases = []
daily_average([daily_cases_sum[0]],avg_daily_cases)
df["Average Daily Cases"] = avg_daily_cases
# daily deaths
deaths = list(df["Total Deaths"])
death_list = []
daily_values(deaths,death_list)
death_list.insert(0,0)
df["Daily Deaths"] = death_list

# average daily deaths
summed_deaths = []
cumulative_sum(death_list,summed_deaths)
avg_deaths = []
daily_average([summed_deaths[0]],avg_deaths)
df["Average Daily Deaths"] = avg_deaths
# Cumulative recovered cases
total = df["Total Cases"]
dead = df["Total Deaths"]
current = df["Active Cases"]
recovered = total - dead - current
df["Recovered"] = recovered

# Daily recovered cases
daily_recovered = []
daily_values(recovered,daily_recovered)
daily_recovered.insert(0,0)
df["Daily Recovered Cases"] = daily_recovered

#Average daily recovered cases
summed_recovered = []
cumulative_sum(daily_recovered,summed_recovered)
avg_recovered = []
daily_average( [summed_recovered[0]], avg_recovered)
df["Average Daily Recovered Cases"] = avg_recovered
df.head()
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.describe()
df.isnull().values.any()

dates = df["Dates"]
total_active_cases = df["Active Cases"]

total_cases = df["Total Cases"]
daily_cases = df["Daily Cases"]
average_daily_cases = df["Average Daily Cases"]

total_deaths = df["Total Deaths"]
daily_deaths = df["Daily Deaths"]
daily_average_deaths = df["Average Daily Deaths"]

total_recovered = df["Recovered"]
daily_recovered = df["Daily Recovered Cases"]
daily_average_recovered_cases = df["Average Daily Recovered Cases"]
# current average is calculated
avg_0 = list(total_cases)
avg = floor( avg_0[-1] / len(avg_0) )

# average is multiplied by number of days in order to draw a horizontal line of average
avg = [avg] * len(dates)

labels = ["Confirnmed Cases", "Confirmed Deaths","Recovered"]
pie_chart_data = [daily_cases.sum(),daily_deaths.sum(), daily_recovered.sum()]
data = go.Pie(labels=labels,values=pie_chart_data)
layout = dict(
title= "Ghana's Covid-19 Cases",title_x=0.5
)

go.Figure(data=data, layout = layout)
style = go.Layout(xaxis={'title':"Days"},yaxis={"title":"Cases"})
fig = go.Figure(layout = style)

trace_0 = fig.add_trace(go.Scatter(x=dates, y=total_cases,name="Cumulative Cases Count"
    ))


fig.update_layout(
    title_text="Ghana's Corona Virus Cumulative Cases", title_x=0.5,
    xaxis=dict(
        showgrid=True,    
        linewidth=2,
        ticks='outside',
        tickangle = -67,
        tickfont=dict(
            family='Arial',
            size=12
        ),
    )
    )

fig.show()
months = [ df.iloc[0:19],
df.loc[20:49],
df.iloc[50:80],
df.iloc[81:110],
df.iloc[111:141],
df.iloc[142:172],
df.iloc[173:-1]]

def monthly(data,array,field):
    for i,j in enumerate(data):
        array.append(data[i][field].sum())


monthly_cases = []
monthly(months,monthly_cases,"Daily Cases")

monthly_recovered = []
monthly(months,monthly_recovered,"Daily Recovered Cases")

monthly_deaths = []
monthly(months,monthly_deaths,"Daily Deaths")


x_axis = ["March", "April","May","June","July","August","September"]
style = go.Layout(xaxis={'title':"Months"},yaxis={"title":"Numbers"})

fig = go.Figure(go.Bar(x=x_axis, y=monthly_cases, name='Monthly Cases'),layout = style)
fig.add_trace(go.Bar(x=x_axis, y=monthly_recovered, name='Monthly Recovered'))
fig.add_trace(go.Bar(x=x_axis, y=monthly_deaths, name='Monthly Deaths'))


fig.update_layout(barmode='stack', title_text="Corona Virus Statistics For Every Month", title_x=0.5,
)
fig.show()
style = go.Layout(xaxis={'title':"Days"},yaxis={"title":"Cases"}, width =850)

fig2 = go.Figure(layout=style)

trace_2 = fig2.add_trace(go.Scatter(x=dates, y=daily_cases,  mode='lines+markers', name="Daily Case Count"
    ))

trace_3 = fig2.add_trace(go.Scatter (x = dates, y=avg, name="Current Average Cases"))
trace_4 = fig2.add_trace(go.Scatter(x=dates, y = average_daily_cases, name="Average Daily Cases"))

fig2.update_layout(
    title_text="Ghana's Daily Corona Virus Cases", title_x=0.5,
    xaxis=dict(
        showgrid=True,    
        linewidth=2,
        ticks='outside',
        tickangle = -67,
        tickfont=dict(
            family='Arial',
            size=12
        ),
    )
    )
fig2.show()
style = go.Layout(width=850)
fig = go.Figure(go.Bar(x=dates, y=daily_recovered, name='Daily Recovered Cases'),layout = style)
fig.add_trace(go.Scatter(x=dates,y=daily_average_recovered_cases, name = "Average Recovered Cases" ))
fig.update_layout(
    title_text="Ghana's Daily Corona Virus Recovered Cases", title_x=0.5,
    xaxis=dict(
        showgrid=True,    
        linewidth=2,
        ticks='outside',
        tickangle = -67,
        tickfont=dict(
            family='Arial',
            size=12
        ),
        
    )
    )
fig.show()
style = go.Layout(width=850)
fig = go.Figure(go.Bar(x=dates, y=daily_deaths, name='Daily Deaths'),layout = style)
fig.add_trace(go.Scatter(x=dates,y=daily_average_deaths, name = "Daily Average Deaths" ))
fig.update_layout(
    title_text="Ghana's Daily Corona Virus Deaths Cases", title_x=0.5,
    xaxis=dict(
        showgrid=True,    
        linewidth=2,
        ticks='outside',
        tickangle = -67,
        tickfont=dict(
            family='Arial',
            size=12
        ),
        
    )
    )
fig.show()


slope, intercept, r, p, std_err = stats.linregress(total_cases, total_deaths)

def regression_line(x):
  return slope * x + intercept

model = list(map(regression_line, total_cases))

trace_1 = go.Scatter(x = total_cases, y=total_deaths, mode='markers', name="Total Cases")
trace_2 = go.Scatter(x=total_cases, y=model, name="Regression Line")
style = go.Layout(yaxis={'title':"Deaths"},xaxis={"title":"Cases"},width = 850)
data = [trace_1,trace_2]

fig = go.Figure(data = data,layout = style)
fig.update_layout(
    title_text=" Linear Regression Model On Ghana's Covid-19 Data", title_x=0.5,
    xaxis=dict(
        showgrid=True,    
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12
        ),
    )
    )
fig.show()

print('Estimated Forecast of total deaths if everyone in Ghana was to contract Covid-19:',
        slope * 30000000 + intercept)
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for{filename}', fontsize=15)
    plt.show()

df.dataframeName = " Ghana's Corona Virus Data"
plotCorrelationMatrix(df, 10)