'''Import Modules'''

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from plotly.offline import plot

from plotly import subplots as t

import plotly.graph_objs as go

import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison);
'''Trace Function'''

def trace (res, color, name):

    trace = go.Scatter(x = res.index[::-1], y = res.values[::-1], name = name, marker = dict(color = color));

    return trace
'''Import Data'''

levelsdata = pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv");

rfdata = pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv");

levelsdata.set_index("Date");

rfdata.set_index("Date");

levelsdata.head()
rfdata.head()
'''Convert to DateTime and then Sort By Date'''

levelsdata["Date"] = pd.to_datetime(levelsdata["Date"], format = '%d-%m-%Y');

rfdata["Date"] = pd.to_datetime(rfdata["Date"], format = '%d-%m-%Y');



# Sort by Date

levelsdata.sort_values(by = 'Date', inplace = True);

rfdata.sort_values(by = 'Date', inplace = True);
'''Extract Water Level Data for Each of the Reservoir'''

poondi = levelsdata["POONDI"];

poondi.index = levelsdata["Date"];

chola = levelsdata["CHOLAVARAM"];

chola.index = levelsdata["Date"];

red = levelsdata["REDHILLS"];

red.index = levelsdata["Date"];

chem = levelsdata["CHEMBARAMBAKKAM"];

chem.index = levelsdata["Date"];

'''Draw Trace'''

poondit = trace(poondi, 'blue', 'Poondi');

cholat = trace(chola, 'orange', 'Cholavaram');

redt = trace(red, 'red', 'Redhills');

chemt = trace(chem, 'purple', ' Chembarambakkam');
'''Calculate the Total Water Level (sum of levels from each reservoir) on a Given Day'''

total = [];

for i in range(0, len(poondi), 1):

    sum  = 0;

    sum = poondi[i] + chola[i] + red[i] + chem[i]

    total.append(sum)

total = pd.Series(total);

total.index = levelsdata["Date"];

totalt = trace(total, 'royalblue', 'Total');

tl ={'Date': total.index, 'total': total.values}

totall2 = pd.DataFrame(data = tl);

totall2['year'] = totall2['Date'].map(lambda x: x.strftime('%Y'));

totall2['modate'] = totall2['Date'].map(lambda x: x.strftime('%m-%d'));

yearwltot = totall2[totall2['modate']=='12-31']; # Will Use Later for calulation of Consumption/Water Loss
'''Water Levels in Each Reservoir'''

levels = t.make_subplots(rows=1, cols=1);

# Reservoir Levels

levels.append_trace(poondit, 1, 1);

levels.append_trace(cholat, 1, 1);

levels.append_trace(redt, 1, 1);

levels.append_trace(chemt, 1, 1);

levels['layout'].update(height = 800, width = 1200, title = "Water Levels (mcft) in Chennai's Four Main Reservoirs", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'))

levels.show();
'''Total Water Levels in all four Reservoir'''

# Total Water Level Only

totfig = go.Figure();

totfig.add_trace(totalt);

totfig['layout'].update(height = 800, width = 1200, title = "Total Water Level (mcft) in Chennai Combined", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'));

totfig.show();
'''Average Water Level'''

mltot = total.resample('Y').mean();

meanwl = go.Figure([go.Bar(x = mltot.index[::-1], y = mltot.values[::-1], marker_color = 'midnightblue', name = 'Average Annual Water Level')]);

meanwl['layout'].update(height = 800, width = 1200, title = "Average Daily Water Level (mcft))", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'));

meanwl.show();
# Post-2018

dip2018 = (levelsdata["Date"] > '2018-01-01');

dip2018levels = levelsdata.loc[dip2018];



poondi = dip2018levels["POONDI"];

poondi.index = dip2018levels["Date"];

chola = dip2018levels["CHOLAVARAM"];

chola.index = dip2018levels["Date"];

red = dip2018levels["REDHILLS"];

red.index = dip2018levels["Date"];

chem = dip2018levels["CHEMBARAMBAKKAM"];

chem.index = dip2018levels["Date"];



poondit = trace(poondi, 'blue', 'Poondi');

cholat = trace(chola, 'orange', 'Cholavaram');

redt = trace(red, 'red', 'Redhills');

chemt = trace(chem, 'purple', ' Chembarambakkam');



totalpo2018 = [];

for i in range(0, len(poondi), 1):

    sum  = 0;

    sum = poondi[i] + chola[i] + red[i] + chem[i]

    totalpo2018.append(sum)

totalpo2018 = pd.Series(totalpo2018);

totalpo2018.index = dip2018levels["Date"];

totalt = trace(totalpo2018, 'royalblue', 'Total');



levels = t.make_subplots(rows=1, cols=1);

levels.append_trace(poondit, 1, 1);

levels.append_trace(cholat, 1, 1);

levels.append_trace(redt, 1, 1);

levels.append_trace(chemt, 1, 1);

levels['layout'].update(height = 800, width = 1200, title = "Water Levels (mcft) in Chennai's Four Main Reservoirs Since 2018", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'))

levels.show();





# Total Water Level Only

totfig = go.Figure();

totfig.add_trace(totalt);

totfig['layout'].update(height = 800, width = 1200, title = "Total Water Level (mcft) in Chennai Combined Since 2018", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Level (mcft)'));

totfig.show();

# Rain Fall

poondirain = rfdata["POONDI"];

poondirain.index = rfdata["Date"];

cholarain = rfdata["CHOLAVARAM"];

cholarain.index = rfdata["Date"];

redrain = rfdata["REDHILLS"];

redrain.index = rfdata["Date"];

chemrain = rfdata["CHEMBARAMBAKKAM"];

chemrain.index = rfdata["Date"];



poondiraint = trace(poondirain, 'blue', 'Poondi');

cholaraint = trace(cholarain, 'orange', 'Cholavaram');

redraint = trace(redrain, 'red', 'Redhills');

chemraint = trace(chemrain, 'purple', ' Chembarambakkam');



totalrain = [];

for i in range(0, len(poondirain), 1):

    sum  = 0;

    sum = poondirain[i] + cholarain[i] + redrain[i] + chemrain[i]

    totalrain.append(sum)

totalrain = pd.Series(totalrain);

totalrain.index = rfdata["Date"];

tr ={'Date': totalrain.index, 'total': totalrain.values}

total2 = pd.DataFrame(data = tr);

total2['year'] = total2['Date'].map(lambda x: x.strftime('%Y'));

yearmax = totalrain.resample('Y').sum();

yearmean = totalrain.resample('Y').mean();

yrt = {'Date':yearmax.index, 'total': yearmax.values};

yearraintot = pd.DataFrame(data = yrt);

yearraintot['year'] = yearraintot['Date'].map(lambda x: x.strftime('%Y'));

yearraintot['modate'] = yearraintot['Date'].map(lambda x: x.strftime('%m-%d'));





# Total Rainfall

totfig = go.Figure();

totfig.add_trace(go.Scatter(x = totalrain.index[::-1], y = totalrain[::-1], mode = 'markers', name = 'Total Rainfall (mm)'));

totfig['layout'].update(height = 800, width = 1200, title = "Total Rainfall (mm) in Chennai Combined", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Rainfall (mm)'));

totfig.show();

# Total Annual Rainfall

totalrain = go.Figure([go.Bar(x = yearmax.index[::-1], y = yearmax.values[::-1], marker_color = 'red', name = 'Total Annual Rain Fall')]);

totalrain['layout'].update(height = 800, width = 1200, title = "Total Annual Rainfall (mm)", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Rainfall (mm)'));

totalrain.show();



# Average Annual Rainfall

meanrain = go.Figure([go.Bar(x = yearmean.index[::-1], y = yearmean.values[::-1], marker_color = 'midnightblue', name = 'Average Annual Rainfall')]);

meanrain['layout'].update(height = 800, width = 1200, title = "Average Annual Rainfall (mm)", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Rainfall (mm)'));

meanrain.show();
# 1-Way ANOVA to see if there is a difference in Average Annual Rain Fall over the time-course of the data

md = ols('total ~ year', data = total2).fit();

anocomp = sm.stats.anova_lm(md, typ =2);

print(anocomp);
# Multi-Comparison to determine which year differs significantly

tuk = MultiComparison(total2['total'], total2['year']);

print(tuk.tukeyhsd())
wl = yearwltot['total'].tolist();

rf = yearraintot['total'].tolist();

con = [];

year = [];

y1 = 2005;

for i in range(0, len(wl) - 1, 1):

    c = (wl[i] + rf[i+1]) - wl[i+1];

    con.append(c);

    year.append(y1);

    y1 += 1;



cr = {'year':year, 'consumption':con};

cr = pd.DataFrame(data = cr);

consumption = go.Figure([go.Bar(x = year, y = con, marker_color = 'red', name = 'Annual Water Consumtion')]);

consumption['layout'].update(height = 800, width = 1200, title = "Annual Water Consumption", xaxis = dict(title = 'Year'), yaxis = dict(title = 'Water Consumption'));

consumption.show();