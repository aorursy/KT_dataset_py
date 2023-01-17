import os
import plotly as py
import plotly.graph_objs as go
from numpy import array, linspace, polyfit
from pandas import read_csv
from collections import Counter
from sklearn.linear_model import LinearRegression as LR
from matplotlib.pyplot import scatter, show, legend, figure, plot
from matplotlib.style import use

#To use ployly offline
py.offline.init_notebook_mode(connected=True)
#print(os.listdir("../input/gun-violence-data_01-2013_03-2018.csv"))
data = read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
length = len(data)
print("{0} data has been loaded!".format(length))
years = []
months = []
states = []
days = []
ages = []
killed_count = 0
injured_count = 0

print("Calculating the required data...")

for i in range(length):
    #Get The Gun Violence Data By Years
    dates = data["date"][i]
    year = dates[0:4]
    years.append(year)
    
    #Get The Gun Violence Data By Month
    dates = data["date"][i]
    month = dates[5:7]
    months.append(month)
    
    #Get The Gun Violence Count In States
    all_states = data["state"][i]
    states.append(all_states)
    
    #Get The Count Of Killed People And Injured People 
    each_killed = data["n_killed"][i]
    killed_count = killed_count + each_killed
    each_injured = data["n_injured"][i]
    injured_count = injured_count + each_injured
    
    #Get The Each Gun Violence Data 
    all_days = data["date"][i]
    day = int(all_days[8:10])
    days.append(day)
    
print("Calculation completed!")
C = Counter(years)

each_years = list(C.keys())
gun_violence_count_of_year = list(C.values())

layout = {
    'title' : 'Gun Violence Data By Years',
    'xaxis' : {
        'title' : 'Years',
    },
    'yaxis' : {
        'title' : 'Gun Violence',
    },
}

trace = go.Bar(
    x = each_years,
    y = gun_violence_count_of_year,
    marker = dict(
      color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )  
    ),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
C = Counter(months)

each_month = list(C.keys())
gun_violence_count_of_month = list(C.values())

layout = {
    'title' : 'Gun Violence Data By Months',
    'xaxis' : {
        'title' : 'Months',
    },
    'yaxis' : {
        'title' : 'Gun Violence',
    },
}

trace = go.Bar(
    x = each_month,
    y = gun_violence_count_of_month,
    marker = dict(
      color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )  
    ),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
C = Counter(states)

each_states = list(C.keys())
gun_violence_count_in_state = list(C.values())

layout = {
    'title' : 'Gun Violence In All States',
    'yaxis' : {
        'title' : 'Gun Violence',
    },
}

trace = go.Bar(
    x = each_states,
    y = gun_violence_count_in_state,
    marker = dict(
      color = 'rgb(158,202,225)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5,
        )  
    ),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
labels = ['Killed People', 'Injured People']
values = [killed_count, injured_count]
colors = ['rgb(8,48,107)', 'rgb(158,202,225)']

layout = {
    'title' : 'Gun Violence In All States',
}

trace = go.Pie(
    labels=labels, 
    values=values,
    textinfo='value',
    textfont=dict(
        size=20, 
        color='rgb(255,255,255)'),
    marker=dict(colors=colors, 
        line=dict(color='rgb(0,0,0)', 
        width=2)),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
C = Counter(days)

each_days = list(C.keys())
gun_violence_count_by_days = list(C.values())

each_days = array(each_days)
gun_violence_count_by_days = array(gun_violence_count_by_days)

s = len(each_days)

each_days = each_days.reshape(s,1)
gun_violence_count_by_days = gun_violence_count_by_days.reshape(s,1)

#linear line drawing and optimal placement
lineerR = LR()
lineerR.fit(each_days,gun_violence_count_by_days)
lineerR.predict(each_days)

#slope calculation
slope = float(lineerR.coef_)
#The intersection point of y for x = 0
interSec = float(lineerR.intercept_)

#finding the lowest and highest values to show the data on the plot
j = sorted(each_days)
#etc. linspace(lowest, highest, frequency)
theta = linspace(j[0], j[s-1], j[0]*j[s-1]*2)

fig = figure(figsize=(18, 9))
use('ggplot')
scatter(
    each_days, 
    gun_violence_count_by_days,
    color="r",
    label="input data")
plot(
    theta, 
    theta * slope + interSec,
    color="b", )
legend()
show()
C = Counter(days)
error = 0.0

each_days = list(C.keys())
gun_violence_count_by_days = list(C.values())

#finding coefficients
a,b,c,d,e,f,g,h = polyfit(each_days,gun_violence_count_by_days, 7)

print("equation of prediction line:\n({0}x^7) + ({1}x^6) + ({2}x^5) + ({3}x^4) + ({4}x^3) + ({5}x^2) + ({6}x) + {7}"
      .format(round(a,4),round(b,4),round(c,4),round(d,4),round(e,4),round(f,4),round(g,4),round(h,4)))

estimation = 5.0

if estimation >= 1 and estimation <= 12:
    result = (a * (estimation ** 7) + b * (estimation ** 6) + c * (estimation ** 5) + d * (estimation ** 4) + 
                e * (estimation ** 3) + f * (estimation ** 2) + g * estimation + h)
    result = round(result, 2)

    print("Real result:\t{0}".format(gun_violence_count_by_days[int(estimation)]))
    print("Estimation result:\t{0}".format(result))
    tolerance = round(abs(result - gun_violence_count_by_days[int(estimation)]),2)
    print("Tolerance of estimated value for month {0}:\t{1}".format(int(estimation), tolerance))

    z = linspace(1,31,350)

    for i in range(len(gun_violence_count_by_days)):
        error = error + abs(gun_violence_count_by_days[i] - (a*(z[i]**7) + b*(z[i]**6) + c*(z[i]**5) + d*(z[i]**4) + 
                                                          e*(z[i]**3) + f*(z[i]**2) + g*(z[i]) + h))

    error = round(error,2)

    print("Total Tolerance:\t{0}".format(error))

    fig = figure(figsize=(20, 10))
    scatter(each_days,gun_violence_count_by_days)
    plot(z, a * (z ** 7) + b * (z ** 6) + c * (z ** 5) + d * (z ** 4) + e * (z ** 3) + f * (z ** 2) + g * z + h, color="g")
    show()        
else:
    print("The number entered is not in the desired range!")