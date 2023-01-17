# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt 

from matplotlib import dates as mdates

from seaborn import heatmap

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')



# Any results you write to the current directory are saved as output.
# Data Cleaning

date_cols = confirmed.select_dtypes('int64').columns



south_korea = confirmed.loc[confirmed['Country/Region'].str.contains('Korea', case=True)]

south_korea = south_korea.max().to_frame().T.set_index('Country/Region')

sk_d = deaths.loc[deaths['Country/Region'].str.contains('Korea', case=True)]

sk_d = sk_d.max().to_frame().T.set_index('Country/Region')



# The other countries I want to look at.

usa = confirmed.loc[confirmed['Country/Region'] == 'US', :].groupby('Country/Region')[date_cols].sum()

usa_d = deaths.loc[deaths['Country/Region'] == 'US', :].groupby('Country/Region')[date_cols].sum()



italy = confirmed.loc[confirmed['Country/Region'] == 'Italy', :].set_index('Country/Region')

italy_d = deaths.loc[confirmed['Country/Region'] == 'Italy', :].set_index('Country/Region')



combined = south_korea.append([usa, italy]).dropna(axis=1)

combined.columns = pd.to_datetime(combined.columns)

combined = combined.T.sort_index()



combined_d = sk_d.append([usa_d, italy_d]).dropna(axis=1)

combined_d.columns = pd.to_datetime(combined_d.columns)

combined_d = combined_d.T.sort_index()
# Additional Dataset (Used as Needed) with some sorting. 

covid_19_all = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")



print(covid_19_all.loc[covid_19_all['Country/Region'].str.contains('Korea', case=True)]['Country/Region'].unique())



# TODO: Convert all these dataframes into a single one with columns based on country, e.g. "US Confirmed" so it's more flexible

addtl_split_data_all = covid_19_all.loc[covid_19_all['Country/Region'].isin(['US', 'Italy', 'South Korea'])]

print(addtl_split_data_all.columns)





#addtl_split_data_all.drop(index=addtl_split_data_all.loc[addtl_split_data_all['Province/State'].str.contains('Princess',na=False)].index)



addtl_us_data_all = addtl_split_data_all.loc[addtl_split_data_all['Country/Region'] == 'US'].groupby('Date').sum()

addtl_us_data_all.index = pd.to_datetime(addtl_us_data_all.index)



addtl_sk_data_all = addtl_split_data_all.loc[addtl_split_data_all['Country/Region'] == 'South Korea'].groupby('Date').sum()

addtl_sk_data_all.index = pd.to_datetime(addtl_sk_data_all.index)



addtl_italy_data_all = addtl_split_data_all.loc[addtl_split_data_all['Country/Region'] == 'Italy'].groupby('Date').sum()

addtl_italy_data_all.index = pd.to_datetime(addtl_italy_data_all.index)



new_columns = ['South Korea', 'US', 'Italy']

addtl_combined = pd.DataFrame(index=addtl_us_data_all.index, columns=new_columns)



addtl_combined['US'] = addtl_us_data_all['Confirmed']

addtl_combined['Italy'] = addtl_italy_data_all['Confirmed']

addtl_combined['South Korea'] = addtl_sk_data_all['Confirmed']



addtl_combined_d = pd.DataFrame(index=addtl_us_data_all.index, columns=new_columns)



addtl_combined_d['US'] = addtl_us_data_all['Deaths']

addtl_combined_d['Italy'] = addtl_italy_data_all['Deaths']

addtl_combined_d['South Korea'] = addtl_sk_data_all['Deaths']



addtl_combined_r = pd.DataFrame(index=addtl_us_data_all.index, columns=new_columns)



addtl_combined_r['US'] = addtl_us_data_all['Recovered']

addtl_combined_r['Italy'] = addtl_italy_data_all['Recovered']

addtl_combined_r['South Korea'] = addtl_sk_data_all['Recovered']



combined = addtl_combined

combined_d = addtl_combined_d

combined_r = addtl_combined_r
# Data set comparisons have been suspended for the moment, so these lines have been commented out. 

# addtl_combined - combined



#print(addtl_split_data_all.loc[addtl_split_data_all['Country/Region'] == 'US']['Province/State'].unique())



#addtl_split_data_all.loc[addtl_split_data_all['Province/State'].str.contains('Princess',na=False)]



#addtl_split_data_all.loc[(addtl_split_data_all['Country/Region'] == 'US') & (addtl_split_data_all['Date'] == '2020-02-22')]

#confirmed[['Country/Region','Province/State', '2/22/20']].loc[(confirmed['Country/Region'] == 'US') & (confirmed['2/22/20'] != 0)]
daily_diff = combined.diff().astype('float64')

daily_2diff = daily_diff.diff().astype('float64')



daily_pct_growth = combined.pct_change() + 1

daily_diff_pct = daily_diff.pct_change()+1

daily_2diff_pct = daily_diff.pct_change()+1

fig, ax = plt.subplots()

combined_d['Italy'].divide(60).plot(ax=ax,logy=True,label="Italy",legend=True,title='Day offset comparison')

combined_d['US'].divide(327).shift(-16).plot(ax=ax,label="US Shifted",legend=True)
fig, ax = plt.subplots(1, 2,figsize=(20,10))

combined.plot(logy=True,ax=ax[0],title='Confirmed, Logarthmic')

combined_d.plot(logy=True,ax=ax[1],title='Deaths, Logarthmic')



fig, ax = plt.subplots(figsize=(20, 20))



daily_pct_growth.tail(30).plot.bar(ax=ax,title='Percentage growth')

ax.axhline(1.0,color='black',ls='dashed')





#daily_diff.plot()

#daily_2diff.plot()

#combined.plot(logy=True)





fig, ax = plt.subplots(figsize=(20,10))

(daily_2diff < 0).rolling(7).sum().tail(28).plot.bar(ax=ax,title='7 day rolling sum of negative 2nd derivatives')
fig_h1, ax_h1 = plt.subplots(2, 3,figsize=(20, 20))



heatmap(daily_diff, ax=ax_h1[0,0])

daily_diff.plot(logy=True,ax=ax_h1[0,1],title='Daily Change Graph, Logarthmic')

daily_diff_pct.tail(14).plot.bar(ax=ax_h1[0,2],title='Percentage change')

ax_h1[0,2].axhline(1.0,color='black',ls='dashed')

#ax_h1[0,2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))





heatmap(daily_2diff, ax=ax_h1[1,0],cmap='seismic',center=0.0)

daily_2diff.plot(ax=ax_h1[1,1],title='Change in Daily Change (2nd derivative)')

daily_2diff_pct.tail(14).plot.bar(ax=ax_h1[1,2],title='Percentage change')

ax_h1[1,2].axhline(1.0,color='black',ls='dashed')







combined_roll = combined.rolling(3, center=True).mean()



fig, ax = plt.subplots()

combined_roll.plot(ax=ax,logy=True,title='Comparison of raw data vs rolling average, Logarthmic')

combined.plot(ax=ax)



daily_diff_roll = combined_roll.diff()

daily_2diff_roll = daily_diff_roll.diff()



fig_h2, ax_h2 = plt.subplots(2, 2,figsize=(20, 20))



heatmap(daily_diff_roll, ax=ax_h2[0,0])

daily_diff_roll.plot(ax=ax_h2[0,1],title='1st Derivative of reported case rolling mean')



heatmap(daily_2diff_roll, ax=ax_h2[1,0],cmap='seismic',center=0.0)

daily_2diff_roll.plot(ax=ax_h2[1,1],title='2nd Derivative of reported case rolling mean')
daily_diff_roll2 = daily_diff.rolling(3, center=True).mean()

daily_2diff_roll2 = daily_diff_roll2.diff()



fig, ax = plt.subplots()

daily_diff.plot(ax=ax, title='Comparison of rolling and raw 1st derivatives')

daily_diff_roll2.plot(ax=ax)



fig_h3, ax_h3 = plt.subplots(1, 2)



heatmap(daily_2diff_roll, ax=ax_h3[0],cmap='seismic',center=0.0)

daily_2diff_roll.plot(ax=ax_h3[1])
combined_d_dt = combined_d.diff().astype('float64')

combined_d_dt2 = combined_d_dt.diff().astype('float64')



fig_h1, ax_h1 = plt.subplots(2, 2,figsize=(20, 20))



heatmap(combined_d_dt, ax=ax_h1[0,0])

combined_d_dt.plot(ax=ax_h1[0,1], title='Fatalities, 1st derivative')



heatmap(combined_d_dt2, ax=ax_h1[1,0],cmap='seismic',center=0.0)

combined_d_dt2.plot(ax=ax_h1[1,1],title='Fatalities, 2nd derivative')
from numpy.fft import fft, fftshift, fftfreq, ifft



fft_data = combined_d_dt.loc['2020-04-01':,'US']

Fk = fftshift(fft(fft_data.to_numpy()))

freqs =  fftshift(fftfreq(fft_data.size,d=1/(fft_data.size))) 



fig, ax = plt.subplots(1, 3,figsize=(20, 10))



ax[0].plot(fft_data.rolling(7,center=True).mean())

ax[0].plot(fft_data)

ax[0].set_title('Daily change in Fatalities')





ax[1].plot(freqs[Fk.real < 20000],Fk.real[Fk.real < 20000])

ax[1].set_title('FFT of daily change, real (removed delta fcn from exponential)')



ax[2].plot(freqs,Fk.imag)

ax[2].set_title('FFT of daily change, imaginary')



print( Fk[(Fk.real < 20000) & (Fk.real > 5000)])

print( freqs[(Fk.real < 20000) & (Fk.real > 5000)] )

fft_data = daily_diff.loc['2020-04-01':,'US']

Fk = fftshift(fft(fft_data.to_numpy()))

freqs =  fftshift(fftfreq(fft_data.size,d=1/(fft_data.size))) 



fig, ax = plt.subplots(1, 3,figsize=(20, 10))



ax[0].plot(fft_data.rolling(7,center=True).mean())

ax[0].plot(fft_data)

ax[0].set_title('Daily change in Confirmed Cases')





ax[1].plot(freqs[Fk.real < 100000],Fk.real[Fk.real < 100000])

ax[1].set_title('FFT of daily change, real (removed delta fcn from exponential)')



ax[2].plot(freqs,Fk.imag)

ax[2].set_title('FFT of daily change, imaginary')



print( Fk[(np.abs(Fk.real) < 100000) & (np.abs(Fk.real) > 20000)])

print( freqs[(np.abs(Fk.real) < 100000) & (np.abs(Fk.real) > 20000)] )
death_pct = combined_d.astype('float64').div(combined.astype('float64'))

death_pct.plot(title='Fatality percentage')

death_pct['2020-02-20':].describe()
def logistic(x,x_0=0,L=1,k=1):

    return L / (1 + np.exp(-k*(x-x_0)))

    
# Calculations

days_to_predict=37



base_mean = death_pct['2020-02-20':]['South Korea'].mean()



predicted_combined = combined_d/base_mean



daily_growth_medians = daily_pct_growth.loc['2020-02-24':'2020-03-24'].median()



growth_array = np.full((days_to_predict,3),daily_growth_medians.to_numpy())

growth_array[0,:] = combined.loc['2020-03-24'].to_numpy()

growth_array = np.cumprod(growth_array,axis=0)



growth_dates = pd.date_range('2020-03-24', periods=days_to_predict, freq='D')

growth_predictions = pd.DataFrame(growth_array, index=growth_dates, columns=combined.columns)



# Technically prediction_error would be a better variable name, but error implies bad, and we want it to be off. 

prediction_difference = combined.loc['2020-03-24':] - growth_predictions





# Logistic function for flat growth rate.

# This doesn't really work. Leaving it in until I figure out how to fix it.

L=357.2e06

k = daily_growth_medians['US']



closest_index = np.searchsorted(growth_predictions['US'],L/2)



x_logi = np.arange(days_to_predict)

y_logi = logistic(x_logi,x_0=closest_index,L=L,k=daily_growth_medians['US'])



logi_predictions = pd.DataFrame(y_logi,index=growth_dates,columns=['US'])



# Curve fitting

y = prediction_difference.loc[:'2020-04-01','US'].to_numpy()

x = np.arange(y.size)



#expfit = np.polyfit(x,np.log(-y+ 0.0001),1)

p_fit = np.polyfit(x,y,4)



#y_e = -np.exp(expfit[0])*np.exp(expfit[1]*x) + 0.0001

y_p = np.poly1d(p_fit)



plt.plot(x,y_p(x))

plt.plot(x,y)



poly_pred_x = np.arange(growth_predictions['US'].size)



poly_pred_base = pd.DataFrame(y_p(poly_pred_x).astype(int), index=growth_predictions.index, columns=['US'])



poly_predictions = poly_pred_base.add(growth_predictions['US'],axis=0)

# Prediction plotting



fig, ax = plt.subplots(1, 2,figsize=(20, 10))

combined['US'].plot(ax=ax[0],legend=True,label='US: Recorded Cases')

predicted_combined['US'].plot(ax=ax[0],legend=True, label='US: Predicted Cases',title='Actual vs Predicted based on mortality rate')



combined['US'].plot(ax=ax[1],legend=True,label='US: Recorded Cases', title='Actual vs Flat Predictions based on mean daily growth rate (logarthmic)')

growth_predictions['US'].plot(ax=ax[1],legend=True,label='US: Flat Growth Predictions',ls='dashed',logy=True)

poly_predictions.plot(ax=ax[1],legend=True,label='US: Polynomial Prediction',ls='dashed')

#logi_predictions.plot(ax=ax[1],legend=True,label='US: Logistic Function (flat growth)', ls='dashed',logy=True)



plt.subplots(figsize=(20,10))

prediction_difference['US'].plot(legend=True, marker='o', title='Difference between recorded cases and flat growth predictions (negative is good)')

fig, ax = plt.subplots(figsize=(20, 10))

combined_r['South Korea'].shift(-25).plot(ax=ax,label='Recovered',legend=True)

combined['South Korea'].plot(ax=ax,label='Recorded Cases',title='Shifted/Lag between Recovered and Recorded cases')
usa_no_NY = addtl_split_data_all.loc[(addtl_split_data_all['Country/Region'] == 'US') & (addtl_split_data_all['Province/State'] != 'New York') ].groupby('Date').sum()

usa_no_NY.index = pd.to_datetime(usa_no_NY.index)



usa_no_NY['Confirmed_dt'] = usa_no_NY['Confirmed'].diff()



fig, ax = plt.subplots(figsize=(20,20))



usa_no_NY['Confirmed_dt'].plot(ax=ax,logy=True,label='US Without NY data',legend=True)

daily_diff['US'].plot(ax=ax,label='US base data',legend=True,title='Daily Differential Cases in US with and without NY data')
fig, ax = plt.subplots(figsize=(20, 10))

combined_r['South Korea'].shift(-20).plot(ax=ax,label='Recovered',legend=True)

combined_d['South Korea'].shift(-15).plot(ax=ax,label='Deaths',legend=True)

combined['South Korea'].plot(ax=ax)
combined_d.shift(10).div(combined_d.shift(10) + combined_r).dropna(axis=0).tail(10)
