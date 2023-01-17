import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np



import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns


pr_data=pd.read_csv("../input/covid19-in-italy/covid19_italy_province.csv", parse_dates=['Date'])

re_data=pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv", parse_dates=['Date'])



re_data.Date=pd.to_datetime(re_data.Date)

print('Columns in provinces database \n',pr_data.columns)

print('Columns in regions database \n',re_data.columns)
pr_data['date'] = pr_data['Date'].dt.date # get ride of hours and minutes

pr_data['date']  = pd.to_datetime(pr_data['date']) # new clean date columnn
latest_date = max(pr_data['date'])

print(latest_date)

province_latest = pr_data[pr_data['date'] == latest_date]
gb_province = province_latest.groupby(['ProvinceName']).sum().reset_index() # get sum of cases for each province

df_province_total = gb_province[['ProvinceName', 'TotalPositiveCases']] # get rid of columns not relevant for this stat

df_province_total = df_province_total.sort_values(by=['TotalPositiveCases'], ascending=False) # sort descending

df_province_total.head()
highest_prov = df_province_total.iloc[0:25]

re_data['date'] = re_data['Date'].dt.date # get ride of hours and minutes

re_data['date']  = pd.to_datetime(re_data['date']) # new clean date columnn
latest_date = max(re_data['date'])

print(latest_date)

region_latest = re_data[re_data['date'] == latest_date]
gb_region = region_latest.groupby(['RegionName']).sum().reset_index() # get sum of cases for each province



df_region_total = gb_region[['RegionName','TestsPerformed', 'TotalPositiveCases', 'HospitalizedPatients', 'IntensiveCarePatients','Recovered', 'Deaths']] # get rid of columns not relevant for this stat

df_region_total = df_region_total.sort_values(by=['TotalPositiveCases'], ascending=False) # sort descending

# Calculation of overall percentages

df_region_total['% positive'] = (df_region_total['TotalPositiveCases']/df_region_total['TestsPerformed'])*100

df_region_total['mortality'] = (df_region_total['Deaths']/df_region_total['TotalPositiveCases'])*100

df_region_total['% hospitalized'] = (df_region_total['HospitalizedPatients']/df_region_total['TotalPositiveCases'])*100

df_region_total['% intensive care'] = (df_region_total['IntensiveCarePatients']/df_region_total['TotalPositiveCases'])*100

df_region_total['% recovered'] = (df_region_total['Recovered']/df_region_total['TotalPositiveCases'])*100

df_region_total.head()
plt.bar(highest_prov.ProvinceName,highest_prov.TotalPositiveCases, color = 'g') #recovered by region

plt.title('Cases by province')

plt.ylabel('Total positive cases')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(18, 6))



plt.subplot(131)

plt.bar(df_region_total.RegionName,df_region_total.TotalPositiveCases, color = 'tomato') #cases by region

plt.title('Active cases by region')

plt.ylabel('Tot Positive Cases ')

plt.xticks(rotation=90)

plt.subplot(132)

plt.bar(df_region_total.RegionName,df_region_total['mortality'], color = 'k') # % deaths by region

plt.title('Mortality by region')

plt.ylabel('% deaths')

plt.xticks(rotation=90)

plt.subplot(133)

plt.bar(df_region_total.RegionName,df_region_total['% recovered'], color = 'b') # provinces with highest num. of cases

plt.title('recovered by region')

plt.ylabel('% recovered')

plt.xticks(rotation=90)

plt.suptitle('Overall selected stats')

plt.show()
print('the average percentage of positive cases over those tested is: ',round(df_region_total['% positive'].mean(), 1),'%') # this is the average of positives detection ove all regions
# quick exploration of the features of the main dataframe

print(re_data.dtypes)

re_data.tail()
gb_time_all = re_data.groupby(['date']).sum().reset_index() # get sum of cases by date



df_time_all = gb_time_all[['date','TotalPositiveCases', 'IntensiveCarePatients', 'TotalHospitalizedPatients', 'HomeConfinement','Deaths', 'Recovered']]

df_time_all['% hospitalized over time'] = (df_time_all['TotalHospitalizedPatients']/df_time_all['TotalPositiveCases'])*100

df_time_all['% deaths over time'] = (df_time_all['Deaths']/df_time_all['TotalPositiveCases'])*100

df_time_all['% intensive care over time'] = (df_time_all['IntensiveCarePatients']/df_time_all['TotalPositiveCases'])*100

df_time_all['% recovered over time'] = (df_time_all['Recovered']/df_time_all['TotalPositiveCases'])*100

plt.figure(figsize=(18, 6))



plt.subplot(121)

plt.plot(df_time_all['date'],df_time_all['% hospitalized over time'], color = 'tomato') #trend hospitalized

plt.title('Hospitalized over time')

plt.ylabel('%hospitalized ')

plt.xticks(rotation=90)





plt.subplot(122)

plt.plot(df_time_all['date'],df_time_all['% intensive care over time'], color = 'b') # provinces with highest num. of cases

plt.title('Intensive care over time')

plt.ylabel('% intensive care')

plt.xticks(rotation=90)

plt.suptitle('Trends hospitalized patients')

plt.show()
plt.figure(figsize=(18, 6))

plt.subplot(121)

plt.plot(df_time_all['date'],df_time_all['% recovered over time'], color = 'g') #trend deaths

plt.title('Recovery over time')

plt.ylabel('% recovered')

plt.xticks(rotation=90)

plt.subplot(122)

plt.plot(df_time_all['date'],df_time_all['% deaths over time'], color = 'k') #trend deaths

plt.title('Mortality over time')

plt.ylabel('% deaths')

plt.xticks(rotation=90)
y = df_time_all['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_y = np.gradient(y) # now we can get the derivative as a new numpy array

#np.savetxt("contagioitalia.csv", deriv_y, delimiter=",")

output = np.transpose(deriv_y)

#now add the numpy array to our dataframe

df_time_all['ContagionRate'] = pd.Series(output)

df_time_all.to_csv('contagioitalia.csv')
#national data fit

from scipy.optimize import curve_fit

from numpy import exp, linspace, random

from math import pi

# build an extrapolated gaussian based on italian data fit

def gaussian(x, amp, cen, wid):

    """1-d gaussian: gaussian(x, amp, cen, wid)"""

    return (amp / (np.sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2)) 

def gauss_function(X, amp, cen, sigma):

    return amp*exp(-(X-cen)**2/(2*sigma**2))



#gaussian modelling for Italy



X1 = df_time_all.index.values

y1 = df_time_all['ContagionRate'].values

#estimate mean and standard deviation

init_vals1 = [2000, 30, 200]  # for [amp, cen, wid]

best_vals1, covar1 = curve_fit(gaussian, X1, y1, p0=init_vals1)



# get scipy values, so we can build an extrapolated gaussian

print('best_vals1: {}'.format(best_vals1))
periods = 80

timerange = pd.date_range(start='2/24/2020', periods=periods)

x_e = np.arange(0, periods)

y_e = gauss_function(x_e, 6000,best_vals1[1],best_vals1[2])
dummy = np.zeros(periods)



plt.figure(figsize=(6, 12))



plt.subplot(211)

plt.plot(df_time_all['date'],df_time_all['TotalPositiveCases'], color = 'g') #trend

plt.plot(timerange,dummy, ':', color = 'w') 

plt.title('Cases over time')

plt.ylabel('number of cases')

plt.xticks(df_time_all['date']," ")

plt.subplot(212)

plt.plot(df_time_all['date'],df_time_all['ContagionRate'], color = 'r') #derivative

#plt.plot(df_time_all['date'],gauss_function(X1, *popt1), ':', label = 'Italy-modelled gaussian')

plt.plot(timerange,y_e, '--', color = 'orange') 

plt.title('Spread rate over time')

plt.ylabel('Rate (number of cases first derivative)')

plt.xticks(rotation=90)

plt.legend()

plt.suptitle('Virus spread over time')

plt.show()
reduced_re_data = re_data[['date','TotalPositiveCases','RegionName']]

#creation of a dataframe for each of the four mainly affected regions

df_piemonte = reduced_re_data[reduced_re_data.RegionName =='Piemonte']

df_lombardia = reduced_re_data[reduced_re_data.RegionName =='Lombardia']

df_veneto = reduced_re_data[reduced_re_data.RegionName =='Veneto']

df_emilia = reduced_re_data[reduced_re_data.RegionName =='Emilia-Romagna']
gb_piemonte = df_piemonte.groupby(['date']).sum().reset_index() # get sum of cases by date

gb_lombardia = df_lombardia.groupby(['date']).sum().reset_index() # get sum of cases by date

gb_veneto = df_veneto.groupby(['date']).sum().reset_index() # get sum of cases by date

gb_emilia = df_emilia.groupby(['date']).sum().reset_index() # get sum of cases by date
gb_piemonte.to_csv('piemonte.csv')

gb_lombardia.to_csv('lombardia.csv')

gb_veneto.to_csv('veneto.csv')

gb_emilia.to_csv('emilia.csv')
yp = gb_piemonte['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_yp = np.gradient(yp) # now we can get the derivative as a new numpy array

output_piemonte = np.transpose(deriv_yp)

#now add the numpy array to our dataframe

gb_piemonte['ContagionRate'] = pd.Series(output_piemonte)

gb_piemonte.head()
yl = gb_lombardia['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_yl = np.gradient(yl) # now we can get the derivative as a new numpy array

output_lombardia = np.transpose(deriv_yl)

#now add the numpy array to our dataframe

gb_lombardia['ContagionRate'] = pd.Series(output_lombardia)

gb_lombardia.head()
yv = gb_veneto['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_yv = np.gradient(yv) # now we can get the derivative as a new numpy array

output_veneto = np.transpose(deriv_yv)

#now add the numpy array to our dataframe

gb_veneto['ContagionRate'] = pd.Series(output_veneto)

gb_veneto.head()
ye = gb_emilia['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_ye = np.gradient(ye) # now we can get the derivative as a new numpy array

output_emilia = np.transpose(deriv_ye)

#now add the numpy array to our dataframe

gb_emilia['ContagionRate'] = pd.Series(output_emilia)

gb_emilia.head()
plt.figure(figsize=(6, 6))

x = gb_piemonte['date']

y = [ gb_piemonte['TotalPositiveCases'], gb_lombardia['TotalPositiveCases'], gb_veneto['TotalPositiveCases'],gb_emilia['TotalPositiveCases'] ]

labels = ['Piemonte', 'Lombardia', 'Veneto', 'Emilia Romagna']



for y_arr, label in zip(y, labels):

    plt.plot(x, y_arr, label=label)

plt.ylabel('Number of cases')

plt.xticks(rotation=90)

plt.title('Cases over time')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.figure(figsize=(6, 6))

x = gb_piemonte['date']

y = [ gb_piemonte['ContagionRate'], gb_lombardia['ContagionRate'], gb_veneto['ContagionRate'],gb_emilia['ContagionRate'] ]

labels = ['Piemonte', 'Lombardia', 'Veneto', 'Emilia Romagna']



for y_arr, label in zip(y, labels):

    plt.plot(x, y_arr, label=label)



plt.title('Spread rate over time')

plt.ylabel('Rate')

plt.xticks(rotation=90)



plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
df_lodi = pr_data[pr_data.ProvinceName =='Lodi']

df_lodi.reset_index(inplace = True, drop=True)

df_torino = pr_data[pr_data.ProvinceName =='Torino']

df_torino.reset_index(inplace = True, drop=True)

df_roma = pr_data[pr_data.ProvinceName =='Roma']

df_roma.reset_index(inplace = True, drop=True)

df_ancona = pr_data[pr_data.ProvinceName =='Ancona']

df_ancona.reset_index(inplace = True, drop=True)

df_milano = pr_data[pr_data.ProvinceName =='Milano']

df_milano.reset_index(inplace = True, drop=True)
ylodi = df_lodi['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_ylodi = np.gradient(ylodi) # now we can get the derivative as a new numpy array

output_lodi = np.transpose(deriv_ylodi)

#now add the numpy array to our dataframe

df_lodi['ContagionRate'] = pd.Series(output_lodi)

df_lodi.tail()
ymilano = df_milano['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_ymilano = np.gradient(ymilano) # now we can get the derivative as a new numpy array

output_milano = np.transpose(deriv_ymilano)

#now add the numpy array to our dataframe

df_milano['ContagionRate'] = pd.Series(output_milano)

df_milano.tail()
yto = df_torino['TotalPositiveCases'].values # transform the column to differentiate into a numpy array

df_torino = df_torino[df_torino.Date !='2020-03-19 17:00:00']# 0 new cases on 19 March close to the peak of epidemy???? 

deriv_yto = np.gradient(yto) # now we can get the derivative as a new numpy array

output_to = np.transpose(deriv_yto)

#now add the numpy array to our dataframe

df_torino['ContagionRate'] = pd.Series(output_to)

df_torino.tail()
yroma = df_roma['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_yroma = np.gradient(yroma) # now we can get the derivative as a new numpy array

output_roma = np.transpose(deriv_yroma)

#now add the numpy array to our dataframe

df_roma['ContagionRate'] = pd.Series(output_roma)

df_roma.tail()
y_ancona = df_ancona['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_y_ancona = np.gradient(y_ancona) # now we can get the derivative as a new numpy array

output_ancona = np.transpose(deriv_y_ancona)

#now add the numpy array to our dataframe

df_ancona['ContagionRate'] = pd.Series(output_ancona)

df_ancona.tail()
plt.figure(figsize=(6, 12))



plt.subplot(211)

plt.plot(df_lodi['Date'],df_lodi['TotalPositiveCases'], color = 'g', label ='Lodi') 

plt.plot(df_torino['Date'],df_torino['TotalPositiveCases'], color = 'orange',  label ='Torino') 

plt.plot(df_roma['Date'],df_roma['TotalPositiveCases'], color = 'b', label ='Roma') 

plt.plot(df_ancona['Date'],df_ancona['TotalPositiveCases'], color = 'k', label ='Ancona') 

plt.plot(df_milano['Date'],df_milano['TotalPositiveCases'], color = 'm', label ='Milano') 

plt.title('Cases over time')

plt.ylabel('number of cases')

plt.xticks(df_time_all['date']," ")

plt.legend()

plt.subplot(212)

plt.plot(df_lodi['Date'],df_lodi['ContagionRate'], color = 'g', label ='Lodi') 

plt.plot(df_roma['Date'],df_roma['ContagionRate'], color = 'b', label = 'Roma') 

plt.plot(df_torino['Date'],df_torino['ContagionRate'], color = 'orange', label = 'Torino')

plt.plot(df_ancona['Date'],df_ancona['ContagionRate'], color = 'k', label ='Ancona') 

plt.plot(df_milano['Date'],df_milano['ContagionRate'], color = 'm', label ='Milano') 

plt.title('Spread rate over time')

plt.ylabel('Rate (daily new cases)')

plt.xticks(rotation=90)

plt.legend()

plt.suptitle('Virus spread over time - provinces',fontsize = 20)

plt.show()