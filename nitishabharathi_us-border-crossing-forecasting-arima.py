import numpy as np 

import pandas as pd 

import matplotlib                 

import matplotlib.pyplot as plt

import matplotlib.gridspec as mgrid

import seaborn as sns              

import calendar 

import plotly.express as px

import plotly.graph_objects as go

import warnings



from statsmodels.tsa.stattools import acf, adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose



warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')

data['Date'] = pd.to_datetime(data['Date'])

data.loc[(data['Port Name'] == 'Eastport') & (data['State'] == 'Idaho'), 'Port Name'] = 'Eastport, ID'

people = data[data['Measure'].isin(['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers'])]

vehicles = data[data['Measure'].isin(['Trucks', 'Rail Containers Full','Truck Containers Empty', 'Rail Containers Empty',

       'Personal Vehicles', 'Buses', 'Truck Containers Full'])]

people_borders = people[['Border','Value']].groupby('Border').sum()

values = people_borders.values.flatten()

labels = people_borders.index



fig = go.Figure(data=[go.Pie(labels = labels, values=values)])

fig.update(layout_title_text='Total inbound persons, since 1996')

fig.show()
p = people[['Date','Border','Value']].set_index('Date')

p = p.groupby([p.index.year, 'Border']).sum()



val_MEX = p.loc(axis=0)[:,'US-Mexico Border'].values.flatten().tolist()

val_CAN = p.loc(axis=0)[:,'US-Canada Border'].values.flatten().tolist()

yrs = p.unstack(level=1).index.values



# Bar chart 

fig = go.Figure(go.Bar(x = yrs, y = val_MEX, name='US-Mexico Border'))

fig.add_trace(go.Bar(x = yrs, y = val_CAN, name='US-Canada Border'))



fig.update_layout(title = 'Total inbounds (people), by border and years', barmode='stack', xaxis={'categoryorder':'category ascending'})

fig.show()
vals = p.unstack().Value

val_MEX = vals['US-Mexico Border']

val_CAN = vals['US-Canada Border']

val_TOT = val_MEX + val_CAN

growth_MEX = val_MEX.diff().dropna()/val_MEX.values[:-1]*100

growth_CAN = val_CAN.diff().dropna()/val_CAN.values[:-1]*100

growth_TOT = val_TOT.diff().dropna()/val_TOT.values[:-1]*100



yrs = vals.index.values

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
m = people[['Date','Measure','Value']].set_index('Date')

m = m.groupby([m.index.year,'Measure']).sum()

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
mb = people[['Date','Border','Measure','Value']].set_index('Date')

mb = mb.groupby([mb.index.year,'Border','Measure']).sum()



fig = go.Figure(data = [go.Bar(x = yrs, y = mb.loc(axis=0)[:,'US-Canada Border', mes].values.flatten().tolist(), name = mes) for mes in measures ])

fig.update_layout(title = 'US-Canada inbounds (people), by measure and years', barmode='stack', xaxis={'categoryorder':'category ascending'})

fig.show()
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
d = data[['Date','Measure','Value']].set_index('Date')

year_measure_df = d.pivot_table('Value', index = d.index.year, columns = 'Measure', aggfunc = 'sum')

year_measure_df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
PStateVehicle_df = people.pivot_table('Value', index = 'State', columns = 'Measure', aggfunc = 'sum')

rest = PStateVehicle_df[PStateVehicle_df.sum(axis=1)  < PStateVehicle_df.sum().sum()*0.04].sum().rename('Rest')



t = PStateVehicle_df[PStateVehicle_df.sum(axis=1)  > PStateVehicle_df.sum().sum()*0.04]

t = t.append(rest)

t = t.iloc[np.argsort(t.sum(axis=1)).values]

t['Other']=t['Bus Passengers'] + t['Train Passengers']

t = t.drop(['Bus Passengers', 'Train Passengers'], axis=1)



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
pcsm = people_crossing_series.loc['2011':]

res_mul = seasonal_decompose(pcsm, model='multiplicative', extrapolate_trend='freq')

res_add = seasonal_decompose(pcsm, model='additive', extrapolate_trend='freq')



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
des = res_mul.trend * res_mul.resid

des.plot(figsize = (15,10))

plt.show()
index_list = des.index

values = list(des)

d = {'Value':values} 

des = pd.DataFrame(d, index = index_list) 

result = adfuller(des.Value.dropna())

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
fig, axes = plt.subplots(3, 2, figsize=(16,10))



axes[0, 0].plot(des.Value)

axes[0, 0].set_title('Original Series')

plot_acf(des, ax=axes[0, 1])



axes[1, 0].plot(des.Value.diff()); axes[1, 0].set_title('1st Order Differentiation')

plot_acf(des.diff().dropna(), ax=axes[1, 1])



axes[2, 0].plot(des.diff().diff()); axes[2, 0].set_title('2nd Order Differentiation')

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



axes[1, 0].plot(des.Value.diff()); axes[1, 0].set_title('1st Order Differencing')

plot_pacf(des.diff().dropna(), ax=axes[1, 1])

axes[2, 0].plot(des.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

plot_pacf(des.diff().diff().dropna(), ax=axes[2, 1])



plt.tight_layout()

plt.show()
model = ARIMA(des, order=(0,1,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(2,2, figsize=(15,8))

residuals.plot(title="Residuals", ax=ax[0,0])

residuals.plot(kind='kde', title='Density', ax=ax[0,1])

plot_acf(model_fit.resid.dropna(), ax=ax[1,0])

plt.tight_layout()

plt.show()
model_fit.plot_predict()

plt.show()
train = des[:74]

test = des[74:]

model_train = ARIMA(train, order=(0,1,1))  



fitted_train = model_train.fit(disp=-1)  

fc, se, conf = fitted_train.forecast(36, alpha=0.05)  # 95% conf

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



plt.figure(figsize=(15,8))

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()