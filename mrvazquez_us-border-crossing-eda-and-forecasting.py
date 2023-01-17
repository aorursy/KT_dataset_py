# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib                  # 2D Plotting Library

import matplotlib.pyplot as plt

import matplotlib.gridspec as mgrid

import seaborn as sns              # Python Data Visualization Library based on matplotlib



import calendar 



### Plotly for interactive plots

import plotly.express as px

import plotly.graph_objects as go



%matplotlib inline
data = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
print(data.info())

data.head()
data.isnull().any()
# Convert dates from strings to date format

data['Date'] = pd.to_datetime(data['Date'])
borders = data['Border'].unique()

print(borders)
years = data['Date'].map(lambda x : x.year).unique()

years
print("There are {} port names.".format(len(data['Port Name'].unique())))

print("There are {} port codes.".format(len(data['Port Code'].unique())))

print("There are {} different locations.".format(len(data['Location'].unique())))
ports = data[['Port Name','Port Code']].drop_duplicates()

ports[ports['Port Name'].duplicated(keep = False)]
data.iloc[[29,217]]
data.loc[(data['Port Name'] == 'Eastport') & (data['State'] == 'Idaho'), 'Port Name'] = 'Eastport, ID'
# take the port's unique code and location

locs = data[['Port Code','Location']]



# take only unique values

locs = locs.drop_duplicates()

print("There are {} different pairs of port code's and locations".format(len(locs)))

# how many locations has each port?

ls = locs['Port Code'].value_counts()

pts = locs['Location'].value_counts()



# Another way of doing the same...

# ls = locs.groupby(['Port Code']).count()

print('')

print('Port codes and # of locations:')

print(ls.head(10))

print('')

print('Locations and # of ports:')

print(pts.head(10))
f,ax = plt.subplots(ncols=2, nrows=2, figsize=(15,10))

sns.set(style = "darkgrid")



# Pie plots

# This function generates autopct, for displaying both the percent value and the original value.

def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct



ls.value_counts().plot.pie(explode = [0.15,0,0],

                           autopct = make_autopct(ls.value_counts().values),

                           ax = ax[0,0])

ax[0,0].set(title = '# of locations per port', ylabel = '')

pts.value_counts().plot.pie(explode = [0.15,0], 

                            autopct = make_autopct(pts.value_counts().values), 

                            ax = ax[1,0])

ax[1,0].set(title = '# of ports per location', ylabel = '')



# Countplots using seaborn

ax[0,1] = sns.countplot(ls, ax=ax[0,1])

ax[0,1].set(title = '# of locations per port', xlabel = '# of locations')

ax[1,1] = sns.countplot(pts, ax=ax[1][1])

ax[1,1].set(title = '# of ports per location', xlabel = '# of ports')



plt.subplots_adjust(

    wspace =  0.25,     # the amount of width reserved for blank space between subplots

    hspace = 0.3 # the amount of height reserved for white space between subplots

)
rpt = pts[pts.values > 1].index

rpt_locs = locs[locs['Location'].isin(rpt)]

rpt_locs
l = rpt_locs.set_index('Location')

pairs = [l.loc[x].values.flatten().tolist() for x in rpt]

print(pairs)
# let's one of those shared locations

data.iloc[[19267,22492]]
data['Measure'].unique()
people = data[data['Measure'].isin(['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers'])]

vehicles = data[data['Measure'].isin(['Trucks', 'Rail Containers Full','Truck Containers Empty', 'Rail Containers Empty',

       'Personal Vehicles', 'Buses', 'Truck Containers Full'])]
people_borders = people[['Border','Value']].groupby('Border').sum()

people_borders
values = people_borders.values.flatten()

labels = people_borders.index

fig = go.Figure(data=[go.Pie(labels = labels, values=values)])

fig.update(layout_title_text='Total inbound persons, since 1996')

fig.show()
# Take the values and set the date as index

p = people[['Date','Border','Value']].set_index('Date')



# Group by years and border

p = p.groupby([p.index.year, 'Border']).sum()

p.head(10)
val_MEX = p.loc(axis=0)[:,'US-Mexico Border'].values.flatten().tolist()

val_CAN = p.loc(axis=0)[:,'US-Canada Border'].values.flatten().tolist()

yrs = p.unstack(level=1).index.values



# Bar chart 

fig = go.Figure(go.Bar(x = yrs, y = val_MEX, name='US-Mexico Border'))

fig.add_trace(go.Bar(x = yrs, y = val_CAN, name='US-Canada Border'))



fig.update_layout(title = 'Total inbounds (people), by border and years', barmode='stack', xaxis={'categoryorder':'category ascending'})

fig.show()
# Unstack the Data Frame

vals = p.unstack().Value

val_MEX = vals['US-Mexico Border']

val_CAN = vals['US-Canada Border']

val_TOT = val_MEX + val_CAN

growth_MEX = val_MEX.diff().dropna()/val_MEX.values[:-1]*100

growth_CAN = val_CAN.diff().dropna()/val_CAN.values[:-1]*100

growth_TOT = val_TOT.diff().dropna()/val_TOT.values[:-1]*100



yrs = vals.index.values



# Bar chart 

# We drop the values for 2019 as there are data only until april

fig = go.Figure(go.Bar(x = yrs, y = growth_MEX.values[:-1], name='US-Mexico Border'))

fig.add_trace(go.Bar(x = yrs, y = growth_CAN.values[:-1], name='US-Canada Border'))

fig.add_trace(go.Line(x = yrs, y = growth_TOT.values[:-1], name='Total'))



fig.update_layout(title = 'Border transit annual growth (people), by border and years', 

                  barmode='group', 

                  xaxis={'categoryorder':'category ascending'},

                  yaxis=go.layout.YAxis(

                      title=go.layout.yaxis.Title(

                      text="Annual growth (%)",

                      font=dict(                      

                      size=18,

                      color="#7f7f7f")

            

        )

    )

                 

                 )

fig.show()
# Take the values and set the date as index

m = people[['Date','Measure','Value']].set_index('Date')



# Group by years and border

m = m.groupby([m.index.year,'Measure']).sum()

m.head(10)
# Bar chart 

measures = ['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers']

yrs = m.unstack().index.values



fig = go.Figure(data = [go.Bar(x = yrs, y = m.loc(axis=0)[:, mes].values.flatten().tolist(), name = mes) for mes in measures ])

    

fig.update_layout(title = 'Total inbounds (people), by measure and years', barmode='stack', xaxis={'categoryorder':'category ascending'})

fig.show()
people_measure = people[['Measure','Value']].groupby('Measure').sum()

values = people_measure.values.flatten()

labels = people_measure.index

fig = go.Figure(data=[go.Pie(labels = labels, values=values)])

fig.update(layout_title_text='Total inbound persons, since 1996')

fig.show()
# Take the values and set the date as index

mb = people[['Date','Border','Measure','Value']].set_index('Date')



# Group by years and border

mb = mb.groupby([mb.index.year,'Border','Measure']).sum()



# Bar chart, US-Canada Border



fig = go.Figure(data = [go.Bar(x = yrs, y = mb.loc(axis=0)[:,'US-Canada Border', mes].values.flatten().tolist(), name = mes) for mes in measures ])

fig.update_layout(title = 'US-Canada inbounds (people), by measure and years', barmode='stack', xaxis={'categoryorder':'category ascending'})

fig.show()
# Bar chart, US-Canada Border



fig = go.Figure(data = [go.Bar(x = yrs, y = mb.loc(axis=0)[:,'US-Mexico Border', mes].values.flatten().tolist(), name = mes) for mes in measures ])

fig.update_layout(title = 'US-Mexico inbounds (people), by measure and years', barmode='stack', xaxis={'categoryorder':'category ascending'})

fig.show()
sns.set(rc={'figure.figsize':(15, 8)})

fig,ax = plt.subplots()

mb.loc(axis=0)[:,'US-Mexico Border', :].unstack().Value.plot(title='US-Mexico Border inbound crossings',ax=ax)

fig.tight_layout()

fig.show()

mb.loc(axis=0)[:,'US-Canada Border', :].unstack().Value.plot(title='US-Canada Border inbound crossings')

plt.show()
# Take the values and set the date as index



start_year = 2014

end_year = 2018



m = people[['Date','Border','Measure','Value']].set_index('Date')



# Group by years and measure

m = m.groupby([m.index.year,'Border', 'Measure']).sum()



m_can = m.loc(axis=0)[start_year:end_year,'US-Canada Border'].groupby('Measure').mean()

m_mex = m.loc(axis=0)[start_year:end_year,'US-Mexico Border'].groupby('Measure').mean()



# plotting, pie charts

f,ax = plt.subplots(ncols=2, nrows=1)



m_can['Value'].plot.pie( ax = ax[0], autopct = '%1.1f%%')

m_mex['Value'].plot.pie( ax = ax[1], autopct = '%1.1f%%')



ax[0].set(title = 'Canadian border, average from {} to {}'.format(start_year,end_year), ylabel = '')

ax[1].set(title = 'Mexican border, average from {} to {}'.format(start_year,end_year), ylabel = '')

f.show()
# Inbounds by years and Measure, since 1996

d = data[['Date','Measure','Value']].set_index('Date')



year_measure_df = d.pivot_table('Value', index = d.index.year, columns = 'Measure', aggfunc = 'sum')

year_measure_df
year_measure_df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
# Incoming people by state and type of vehicle, since 1996



PStateVehicle_df = people.pivot_table('Value', index = 'State', columns = 'Measure', aggfunc = 'sum')

PStateVehicle_df
rest = PStateVehicle_df[PStateVehicle_df.sum(axis=1)  < PStateVehicle_df.sum().sum()*0.04].sum().rename('Rest')



t = PStateVehicle_df[PStateVehicle_df.sum(axis=1)  > PStateVehicle_df.sum().sum()*0.04]

t = t.append(rest)

# Sort them by total flux

t = t.iloc[np.argsort(t.sum(axis=1)).values]

# Combine Train and bus passagners in "others"

t['Other']=t['Bus Passengers'] + t['Train Passengers']

t = t.drop(['Bus Passengers', 'Train Passengers'], axis=1)



# Plot

fig, ax = plt.subplots()



size = 0.4



a= t.sum(axis=1).plot.pie(radius = 1,

       wedgeprops=dict(width=size+0.23, edgecolor='w'), ax = ax, autopct = '%1.1f%%', pctdistance= 0.8)



b=pd.Series(t.values.flatten()).plot.pie(radius = 1- size,colors = ['#DF867E','#8DC0FB','#A9EE84'],

       wedgeprops=dict(width=size-0.2, edgecolor='w'), ax=ax, labels = None)



ax.set(ylabel=None)

red_patch = matplotlib.patches.Patch(color='#DF867E', label='Pedestrians')

blue_patch = matplotlib.patches.Patch(color='#8DC0FB', label='Personal vehicle passengers')

green_patch = matplotlib.patches.Patch(color='#A9EE84', label='Others')

plt.legend(handles=[blue_patch,red_patch, green_patch], loc='best', bbox_to_anchor=(0.75, 0.5, 0.5, 0.5))



plt.show()
start_year = 2015

end_year = 2018



# Group by years and states

p_states = people[['Date','State','Value']].set_index('Date')

p_states = p_states.groupby([p_states.index.year, 'State']).sum()

# Select date range and compute mean

p_states = p_states.loc(axis=0)[start_year:end_year,:].groupby('State').mean()

# Sort, for nice visualization

p_states = p_states['Value'].sort_values()

# Take only states with more than 4% of share 

rest = p_states[p_states < p_states.sum()*.04].sum()

p_states = p_states[p_states > p_states.sum()*.04].append(pd.Series({'Rest' : rest}))



# Same for all years:

p_states_tot = people[['State','Value']].groupby('State').sum()

p_states_tot = p_states_tot['Value'].sort_values()

rest_tot = p_states_tot[p_states_tot < p_states_tot.sum()*.04].sum()

p_states_tot = p_states_tot[p_states_tot > p_states_tot.sum()*.04].append(pd.Series({'Rest' : rest_tot}))





# plotting, pie charts

f,ax = plt.subplots(ncols=2, nrows=1)



p_states_tot.plot.pie( ax = ax[0], autopct = '%1.1f%%')

p_states.plot.pie( ax = ax[1], autopct = '%1.1f%%')



ax[0].set(title = 'States share (inbound people), since 1996', ylabel = '')

ax[1].set(title = 'States share (inbound people), average from {} to {}'.format(start_year,end_year), ylabel = '')

f.show()
p_ports = people[['Port Name','Value']].groupby('Port Name').sum().Value.sort_values(ascending = False)

p_ports.hist()



plt.show()
num_p = 10



pctg = p_ports.head(num_p).sum()/p_ports.sum()*100



print('The {} most transited ports are:'.format(num_p))

print(p_ports.head(num_p))

print('')

print ('The {} most transited ports (out of 117) take {:.2f} % of all persons crossings into the US.'.format(num_p, pctg))
p_locs = people[['Location','Value']].groupby('Location').sum().reset_index()



# functions to get coordinates,

def get_lon(point) :

    par = point[7:-1].partition(' ')

    return float(par[0])



def get_lat(point) :

    par = point[7:-1].partition(' ')

    return float(par[2])



# adding a column with the coordinates to the dataframe

p_locs['lat'] = p_locs.Location.apply(lambda x: get_lat(x))

p_locs['lon'] = p_locs.Location.apply(lambda x: get_lon(x))



# As some locations have 2 ports, we have to take are of it in the labelling,

ps = data[['Port Name','Location']].drop_duplicates().set_index('Location')

p_locs['Ports'] = p_locs.Location.apply(lambda x : ', '.join(ps.loc[x].values.flatten()))

p_locs['text'] = p_locs['Ports'] + '<br>Crossings: ' + (p_locs['Value']/1e6).astype(str)+' million'





color = "crimson"

scale = 500000



fig = go.Figure()

fig.add_trace(go.Scattergeo(

    locationmode = 'USA-states',

    lon = p_locs['lon'],

    lat = p_locs['lat'],

    text = p_locs['text'],

    marker = dict(

        size = p_locs['Value']/scale,

        color = color,

        line_color='rgb(40,40,40)',

        line_width=0.5,

        sizemode = 'area')))



fig.update_layout(

        title_text = 'US Borders, total inbound persons since 1996<br>(Click legend to toggle traces)',

        showlegend = False,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    )



fig.show()
people_crossing_series = people[['Date','Value']].groupby('Date').sum()

people_crossing_series_CAN = people[people['Border'] == 'US-Canada Border'][['Date','Value']].groupby('Date').sum()

people_crossing_series_MEX = people[people['Border'] == 'US-Mexico Border'][['Date','Value']].groupby('Date').sum()
sns.set(rc={'figure.figsize':(15, 8)})

fig, ax = plt.subplots()



#Define a rolling mean, by years

rmean = people_crossing_series.rolling(12, center=True).mean()

rmean_MEX = people_crossing_series_MEX.rolling(12, center=True).mean()

rmean_CAN = people_crossing_series_CAN.rolling(12, center=True).mean()



ax.plot(people_crossing_series,

       marker='.', linestyle='-', linewidth=1, alpha = 1, label='Total')

ax.plot(rmean,

       marker=None, linestyle='-', linewidth=1.5, alpha = 0.5, label='Total, rolling mean (years)', color = 'b')



ax.plot(people_crossing_series_MEX,

       marker='.', linestyle='-', linewidth=1, alpha = 1, label='Mexico', color = 'r')

ax.plot(rmean_MEX,

       marker=None, linestyle='-', linewidth=1.5, alpha = 0.5, label='Mexico, rolling mean (years)', color = 'r')



ax.plot(people_crossing_series_CAN,

       marker='.', linestyle='-', linewidth=1, alpha = 1, label='Canada', color = 'g')

ax.plot(rmean_CAN,

       marker=None, linestyle='-', linewidth=1.5, alpha = 0.5, label='Canada, rolling mean (years)', color = 'g')



ax.set(title = 'Total monthly persons entering in the US, from 1996', xlabel = 'year')

ax.legend()



plt.show()
fig, ax = plt.subplots()



start = '2015'

end = '2018'





ax.plot(people_crossing_series.loc[start:end],

       marker='o', linestyle='-', linewidth=0.8, alpha = 1, label='Total', color = 'b')

ax.plot(rmean.loc[start:end],

       marker=None, linestyle='-', linewidth=1.5, alpha = 0.5, label='Total, rolling mean (years)', color = 'b')



ax.plot(people_crossing_series_MEX.loc[start:end],

       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Mexico', color = 'r')

ax.plot(rmean_MEX.loc[start:end],

       marker=None, linestyle='-', linewidth=1.5, alpha = 0.5, label='Mexico, rolling mean (years)', color = 'r')



ax.plot(people_crossing_series_CAN.loc[start:end],

       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Canada',color = 'g')

ax.plot(rmean_CAN.loc[start:end],

       marker=None, linestyle='-', linewidth=1.5, alpha = 0.5, label='Canada, rolling mean (years)', color = 'g')



ax.set(title = 'Total persons entering in the US, from {} to {}'.format(start, end))

ax.legend()



plt.show()
fig = plt.figure()



grid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])



seas = fig.add_subplot(grid[0])

trend = fig.add_subplot(grid[1], sharex = seas)



start = '2015'

end = '2018'



seas.plot(people_crossing_series.loc[start:end]/people_crossing_series.loc[start:end].sum(),

       marker='o', linestyle='-', linewidth=0.8, alpha = 1, label='Total', color = 'b')



seas.plot(people_crossing_series_MEX.loc[start:end]/people_crossing_series_MEX.loc[start:end].sum(),

       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Mexico', color = 'r')



seas.plot(people_crossing_series_CAN.loc[start:end]/people_crossing_series_CAN.loc[start:end].sum(),

       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Canada', color = 'g')



seas.set(title = 'Persons entering in the US, from {} to {}, normalised'.format(start, end),

      ylabel = 'arbitrary units')

seas.legend()



trend.plot(rmean.loc[start:end]/rmean.loc[start:end].sum(),

       marker='', linestyle='-', linewidth=2, alpha = 1, label='Total', color = 'b')



trend.plot(rmean_MEX.loc[start:end]/rmean_MEX.loc[start:end].sum(),

       marker='', linestyle='-', linewidth=2, alpha = 1, label='Mexico', color = 'r')



trend.plot(rmean_CAN.loc[start:end]/rmean_CAN.loc[start:end].sum(),

       marker='', linestyle='-', linewidth=2, alpha = 1, label='Canada', color = 'g')



trend.set(ylabel = ' Trend (arbitrary units)')

fig.tight_layout()

plt.show()
start = '2011'

end = '2018'

pcsm = people_crossing_series.loc[start:end]



fig, ax = plt.subplots(2,figsize = (18,13))



for i in range(11) :

    mm = pcsm[pcsm.index.month == i] 

    ax[0].plot(mm, label = calendar.month_abbr[i])

    ax[1].plot(mm/mm.sum(), label = calendar.month_abbr[i])

    

ax[0].set(title = 'persons entering the US between {} and {}, total by months'.format(start, end),

         ylabel = '# people')

ax[1].set(title = 'persons entering the US between {} and {}, trend by months'.format(start, end),

         ylabel = 'arbitrary units')

ax[0].legend()

ax[1].legend()



plt.show()
start = '2011'

end = '2018'

pcsm = people_crossing_series.loc[start:end]

months = [calendar.month_abbr[m] for m in range(1,13)]

fig, ax = plt.subplots(2,figsize = (18,13))



start = int(start)

end = int(end)



for i in range(start, end) :

    yy = pcsm[pcsm.index.year == i];

    yy = yy.set_index(yy.index.month);

    ax[0].plot(yy

               , label = i)

    ax[1].plot(yy/yy.sum()

               , label = i)

    

ax[0].set(title = 'persons entering the US between {} and {}, total by years'.format(start, end),

         ylabel = '# people')



ax[1].set(title = 'persons entering the US between {} and {}, seasonal (normalised)'.format(start, end),

         ylabel = 'arbitrary units')



plt.setp(ax, xticks = range(1,13), xticklabels = months)

ax[0].legend()

plt.tight_layout()



plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose



pcsm = people_crossing_series.loc['2011':]



# Multiplicative Decomposition 

res_mul = seasonal_decompose(pcsm, model='multiplicative', extrapolate_trend='freq')



# Additive Decomposition

res_add = seasonal_decompose(pcsm, model='additive', extrapolate_trend='freq')



# extrapolate_trend='freq' gets rid of NaN values
# Plot

fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(15,8))



res_mul.observed.plot(ax=axes[0,0], legend=False)

axes[0,0].set_ylabel('Observed')



res_mul.trend.plot(ax=axes[1,0], legend=False)

axes[1,0].set_ylabel('Trend')



res_mul.seasonal.plot(ax=axes[2,0], legend=False)

axes[2,0].set_ylabel('Seasonal')



res_mul.resid.plot(ax=axes[3,0], legend=False)

axes[3,0].set_ylabel('Residual')



res_add.observed.plot(ax=axes[0,1], legend=False)

axes[0,1].set_ylabel('Observed')



res_add.trend.plot(ax=axes[1,1], legend=False)

axes[1,1].set_ylabel('Trend')



res_add.seasonal.plot(ax=axes[2,1], legend=False)

axes[2,1].set_ylabel('Seasonal')



res_add.resid.plot(ax=axes[3,1], legend=False)

axes[3,1].set_ylabel('Residual')



axes[0,0].set_title('Multiplicative')

axes[0,1].set_title('Additive')

    

plt.tight_layout()

plt.show()
# Deseasonalized data



des = res_mul.trend * res_mul.resid

des.plot(figsize = (15,10))



plt.show()
# Checking the stationarity through a ADF test statistics



from statsmodels.tsa.stattools import adfuller



result = adfuller(des.Value.dropna())

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
import numpy as np, pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt

#plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



# Original Series

fig, axes = plt.subplots(3, 2, figsize=(16,10))



axes[0, 0].plot(des.Value)

axes[0, 0].set_title('Original Series')

plot_acf(des, ax=axes[0, 1])



# 1st Differencing

axes[1, 0].plot(des.Value.diff()); axes[1, 0].set_title('1st Order Differencing')

plot_acf(des.diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(des.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

plot_acf(des.diff().diff().dropna(), ax=axes[2, 1])



plt.tight_layout()

plt.show()
result_diff = adfuller(des.diff().Value.dropna())

print('ADF Statistic: %f' % result_diff[0])

print('p-value: %f' % result_diff[1])
fig, axes = plt.subplots(3, 2, figsize=(16,10))



axes[0, 0].plot(des.Value)

axes[0, 0].set_title('Original Series')

plot_pacf(des, ax=axes[0, 1])



# 1st Differencing

axes[1, 0].plot(des.Value.diff()); axes[1, 0].set_title('1st Order Differencing')

plot_pacf(des.diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(des.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

plot_pacf(des.diff().diff().dropna(), ax=axes[2, 1])



plt.tight_layout()

plt.show()
from statsmodels.tsa.arima_model import ARIMA



# ARIMA(p,d,q) Model 

model = ARIMA(des, order=(0,1,1))

model_fit = model.fit(disp=0)



# Print the information of the model

print(model_fit.summary())
# Plot residual errors



residuals = pd.DataFrame(model_fit.resid)



fig, ax = plt.subplots(2,2, figsize=(15,8))

residuals.plot(title="Residuals", ax=ax[0,0])

residuals.plot(kind='kde', title='Density', ax=ax[0,1])

plot_acf(model_fit.resid.dropna(), ax=ax[1,0])

plt.tight_layout()

plt.show()
# Actual vs Fitted

model_fit.plot_predict()

plt.show()
print(len(des))

print(len(des)*.75)
from statsmodels.tsa.stattools import acf



# Create Training and Test

train = des[:74]

test = des[74:]



# Build Model

model_train = ARIMA(train, order=(0,1,1))  

# and fit- using MLE method

fitted_train = model_train.fit(disp=-1)  



# Forecast

fc, se, conf = fitted_train.forecast(25, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(15,8))

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm



model_auto = pm.auto_arima(des, start_p=0, start_q=0,

                      test='adf',       # use adftest to find optimal 'd'

                      max_p=4, max_q=4, # maximum p and q

                      m=12,              # frequency of series

                      d=None,           # let model determine 'd'

                      seasonal=False,   # No Seasonality

                      start_P=0, 

                      D=0, 

                      trace=True,

                      error_action='ignore',  

                      suppress_warnings=True, 

                      stepwise=False)



print(model_auto.summary())
model_auto.plot_diagnostics(figsize=(15,8))

plt.show()
# What's the last date in our data?

people_crossing_series.tail(1)
# Intervals for forecasting



date_start = people_crossing_series.tail(1).index[0]

date_end = '2020-12-01'

date_rng = pd.date_range(start=date_start, end=date_end, freq='MS', closed = 'right') # range for forecasting

n_forecast = len(date_rng) # number of steps to forecast



seasonal = res_mul.seasonal.loc['2018-01-01':'2018-12-01'].values # seasonal component, we take the 2018 ones, but they are all the same.

tms = pd.Series(np.tile(seasonal.flatten(),11), index = pd.date_range(start='2019-01-01', end = '2029-12-01', freq='MS'))  # This is just a very long series with the seasonality.



def make_seasonal(ser) :

    seasonal_series = ser * tms # Include the seasonality

    seasonal_series = seasonal_series[~seasonal_series.isnull()] # trim extra values

    return seasonal_series

    

# Forecast



model = ARIMA(des, order=(0,1,1))

model_fit = model.fit(disp=0)



fc1, se1, conf1 = model_fit.forecast(n_forecast, alpha = 0.0455)  # 2 sigma Confidence Level (95,55% conf)

fc2, se2, conf2 = model_fit.forecast(n_forecast, alpha = 0.3173)  # 1 sigma Confidence Level (68,27% conf)



# Make as pandas series 

fc1_series = pd.Series(fc1, index = date_rng)

lower_series1 = pd.Series(conf1[:, 0], index = date_rng)

upper_series1 = pd.Series(conf1[:, 1], index = date_rng)



# Include seasonality

fc1_series, lower_series1, upper_series1 = [make_seasonal(fc1_series), make_seasonal(lower_series1), make_seasonal(upper_series1)]



plt.figure(figsize=(12,5), dpi=100)



#plt.plot(des, label='actual')

#plt.plot(people_crossing_series, label='actual')

plt.plot(des * res_mul.seasonal, label='data')

plt.plot(fc1_series , label='forecast')



# Confidence level intervals

plt.fill_between(lower_series1.index,lower_series1, upper_series1, 

                 color='k', alpha=.15, label='2$\sigma$ Confidence level (95%)')

plt.title('Forecast 2019/20')

plt.legend(loc='upper left', fontsize=8)

#plt.ylim(10000000, 30000000)

plt.xlim('2016', '2021')

plt.show()
fig = plt.figure()



grid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])

sns.set(rc={'figure.figsize':(15, 8)})



seas = fig.add_subplot(grid[0])

seas_adj = fig.add_subplot(grid[1], sharex = seas)



seas.plot(des * res_mul.seasonal, label='data')

seas.plot(fc1_series , label='forecast')

seas.fill_between(lower_series1.index,lower_series1, upper_series1, 

                 color='k', alpha=.15, label = '2$\sigma$ Confidence level (95%)')



# seasonal adjusted:

seas_adj.plot(des, label='data')



seas_adj.plot(pd.Series(fc1, index = date_rng) , label='forecast')

seas_adj.fill_between(date_rng,

                  pd.Series(conf1[:, 0], index = date_rng), 

                  pd.Series(conf1[:, 1], index = date_rng), 

                 color='k', alpha=.15, label = '2$\sigma$ Confidence level (95%)')

plt.xlim('2016', '2021')

seas.set_title('Forecast 2019/20')

seas.legend()

fig.tight_layout()



plt.show()