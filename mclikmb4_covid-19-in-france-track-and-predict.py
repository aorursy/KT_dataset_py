import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



raw_data = pd.read_csv( "../input/coronavirusdataset-france/chiffres-cles.csv", parse_dates=['date']) # raw dataframe

df_china = pd.read_csv( "../input/coronavirusdataset-france/china.csv", parse_dates=['date'])

df_italy = pd.read_csv( "../input/coronavirusdataset-france/contagioitalia.csv", parse_dates=['date'])

df_korea = pd.read_csv( "../input/coronavirusdataset/Case.csv")

df_korea_trend = pd.read_csv( "../input/coronavirusdataset/SearchTrend.csv", parse_dates=['date'])

df_korea_time = pd.read_csv( "../input/coronavirusdataset/Time.csv", parse_dates=['date'])

df_korea_patient = pd.read_csv( "../input/coronavirusdataset/PatientInfo.csv")
df_italy.tail()
raw_data.rename(columns={'cas_confirmes':'cases', 'deces':'deaths'},inplace=True) #important variable names in English

# here I check that stats are updated to the latest governamental info

latest_date = max(raw_data['date'])

print("Stats updated to:",latest_date)

national_latest = raw_data[raw_data['date'] == latest_date]



df_national = raw_data[raw_data.maille_nom =='France']

df_national.tail()




df_national.reset_index(inplace = True, drop=True)

df_national = df_national[['date','cases','deaths', 'nouvelles_hospitalisations', 'nouvelles_reanimations']]

df_national = df_national.groupby(['date']).mean().reset_index() # get cases each day
df_national = df_national[df_national['date'] > '2020-03-01']

df_national.date = pd.to_datetime(df_national.date)

df_national.reset_index(inplace = True, drop=True)

df_national = df_national.drop([56, 57])

df_national = df_national.dropna()

df_national.reset_index(inplace = True, drop=True)

df_national.tail()
y = df_national['cases'].values # transform the column to differentiate into a numpy array



deriv_y = np.gradient(y) # now we can get the derivative as a new numpy array



output = np.transpose(deriv_y)

#now add the numpy array to our dataframe

df_national['ContagionRate'] = pd.Series(output)

df_national.to_csv('contagiofrancia.csv')
periods = 115



timerange = pd.date_range(start='3/1/2020', periods=periods)

dummy = np.zeros(periods)

plt.figure(figsize= (6,12))

plt.subplot(211)

plt.plot(df_national['date'],df_national['cases'], color = 'g') #trend cases

plt.plot(timerange,dummy, ':', color = 'w') 

plt.title('Cases over time')

plt.ylabel('number of cases')

plt.xticks(df_national['date']," ")

plt.subplot(212)

plt.plot(df_national['date'],df_national['ContagionRate'], color = 'r', label = 'new cases') #trend daily cases



plt.title('Spread rate over time')

plt.ylabel('Rate (new cases per day)')

plt.legend()

plt.xticks(rotation=90)



plt.suptitle('Virus spread over time - France', fontsize=16)

plt.show()
dummy = np.zeros(periods)

plt.figure(figsize= (6,12))

plt.subplot(211)

plt.plot(df_national['date'],df_national['nouvelles_hospitalisations'], color = 'k', label = 'new cases') #trend cases

plt.plot(timerange,dummy, ':', color = 'w') 

plt.title('Hospital cases')

plt.ylabel('new patients in hospital')

plt.xticks(df_national['date']," ")

plt.subplot(212)

plt.plot(df_national['date'],df_national['nouvelles_reanimations'], color = 'r', label = 'new cases') #trend daily cases

 

plt.title('Intensive care')

plt.ylabel('new patients in intensive care')

plt.legend()

plt.xticks(rotation=90)



plt.suptitle('Virus spread over time - France', fontsize=16)

plt.show()
yc = df_china['Number of cases'].values # transform the column to differentiate into a numpy array



deriv_yc = np.diff(yc) # now we can get the derivative as a new numpy array

output_c = np.transpose(deriv_yc)



df_china['ContagionRate'] = pd.Series(output_c) # 



df_china = df_china[df_china['ContagionRate'] < 4500] # clean the chinese data from the suspicious "spike" of 12/2
y_it = df_italy['TotalPositiveCases'].values # transform the column to differentiate into a numpy array



deriv_y_it = np.gradient(y_it) # now we can get the derivative as a new numpy array

#np.savetxt("contagioitalia.csv", deriv_y, delimiter=",")

output_it = np.transpose(deriv_y_it)

#now add the numpy array to our dataframe

df_italy['ContagionRate'] = pd.Series(output_it)
y_kr = df_korea_time['confirmed'].values # transform the column to differentiate into a numpy array



deriv_y_kr = np.gradient(y_kr) # now we can get the derivative as a new numpy array

#np.savetxt("contagioitalia.csv", deriv_y, delimiter=",")

output_kr = np.transpose(deriv_y_kr)

#now add the numpy array to our dataframe

df_korea_time['ContagionRate'] = pd.Series(output_kr)
X_ch = df_china.index.values



y_ch = df_china['ContagionRate'].values



#print(len(X_ch), len(y_ch))
population_china = 1427647786 

population_italy = 60488373

population_france = 65241316
plt.figure(figsize=(12, 10))

plt.subplot(221)

plt.plot(df_national.index,df_national['cases'], label = 'France') #trend cases

plt.plot(df_italy.index,df_italy['TotalPositiveCases'], label = 'Italy') #trend cases

plt.plot(df_china.index, df_china['Number of cases'], label = 'China') #trend cases

plt.title('International comparison of cases growth', fontsize = 20)

plt.xlabel('Days', fontsize=14)

plt.ylabel('Num. cases', fontsize=14)



plt.subplot(222)

plt.plot(df_national.index,(df_national['cases']/population_france)*100, label = 'France') #trend cases

plt.plot(df_italy.index,(df_italy['TotalPositiveCases']/population_italy)*100, label = 'Italy') #trend cases

plt.plot(df_china.index, (df_china['Number of cases']/population_china)*100, label = 'China') #trend cases

plt.xlabel('Days', fontsize=14)

plt.yscale('log')

plt.ylabel('Log %cases over population total', fontsize=14)





plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.figure(figsize= (6,6))

#plot the fit results

#plt.plot(X_ch,gauss_function(X_ch, *popt), ':', label = 'China-modelled gaussian')

#plt.plot(X1,y1, ':', label = 'Italy-modelled gaussian')

#plt.plot(X,gauss_function(X, *popt2), '--', label = 'France-modelled gaussian')

#confront with the given data

plt.plot(df_national.index,df_national['ContagionRate'], label = 'France') #trend cases

plt.plot(df_italy.index,df_italy['nuovi_positivi'], label = 'Italy') #trend cases

plt.plot(df_china.index, df_china['ContagionRate'], label = 'China') #trend cases

plt.axvline(x=33 , color='k', linewidth = 0.5)

plt.axvline(x=59, ymin=0.05, ymax=0.4, color='k', linewidth = 0.5)

plt.text(35, 6000, ' Italy: 2020-03-28\n France: 2020-04-04')

plt.text(59, 3000, ' Italy: 2020-04-24\n France: 2020-05-01')

plt.title('Cases over time')

plt.ylabel('Spread rate')



plt.xticks(rotation=90)

plt.xlim(0,300)

plt.ylabel('Contagion rate (daily new infections)', fontsize=14)

plt.xlabel('Days', fontsize = 14)

plt.legend



plt.title('International comparison of spread rate', fontsize = 20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# clean up noise from low values

df_italy_clean = df_italy[df_italy['ContagionRate'] > 100] 

df_france_clean = df_national[df_national['ContagionRate'] > 100] 

df_china_clean = df_china[df_china['ContagionRate'] > 100] 

df_korea_clean = df_korea_time[df_korea_time['ContagionRate'] > 100]
plt.figure(figsize=(6, 6))



plt.plot(df_france_clean['cases'], df_france_clean['ContagionRate'], label ='France')

plt.plot(df_italy_clean['TotalPositiveCases'], df_italy_clean['ContagionRate'], label = 'Italy')

plt.plot(df_china_clean['Number of cases'], df_china_clean['ContagionRate'], label = 'China')

plt.plot(df_korea_clean['confirmed'], df_korea_clean['ContagionRate'], label = 'Korea')



plt.xscale('log')

plt.yscale('log')

plt.ylabel('Daily new cases', fontsize=14)

plt.xlabel('Total cases', fontsize = 14)

plt.legend



plt.title('International comparison of contagion rates', fontsize = 20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)





plt.show()
df_departement =  raw_data[raw_data.granularite =='departement']

df_departement.rename(columns={'maille_nom':'district'},inplace=True) 

df_departement.tail()
gb_departement =   df_departement[df_departement['date'] == df_departement['date'].max()].reset_index() # get sum of cases by district

gb_departement.rename(columns={'cases':'TotalPositiveCases'},inplace=True) 



gb_departement['mortality'] = (gb_departement['deaths']/gb_departement['hospitalises'])*100

gb_departement['recovered'] = (gb_departement['gueris']/gb_departement['hospitalises'])*100



gb_departement = gb_departement.sort_values(by=['hospitalises'], ascending=False).reset_index() # sort descending

gb_departement.head()
plt.figure(figsize=(12, 12))



plt.subplot(313)

plt.bar(gb_departement.district.iloc[0:35],gb_departement.hospitalises.iloc[0:35], color = 'tomato') #cases by region

plt.title('Hospitalized by district', fontsize = 14)

plt.ylabel('People in hospital')

plt.xticks(rotation=90)



plt.subplot(312)

plt.bar(gb_departement.district.iloc[0:35],gb_departement['gueris'].iloc[0:35], color = 'g') # % deaths by region

plt.title('Recovered by district', fontsize = 14)

plt.ylabel('Recovered')

plt.xticks(gb_departement['district'].iloc[0:35]," ")





plt.subplot(311)

plt.bar(gb_departement.district.iloc[0:35],gb_departement['deaths'].iloc[0:35], color = 'k') # % deaths by region



plt.ylabel('Deaths')

plt.xticks(gb_departement['district'].iloc[0:35]," ")

plt.title('Mortality by district', fontsize = 14)



plt.suptitle('Overall departemental stats', fontsize = 20)

plt.show()
df_region = raw_data[raw_data.granularite == 'region']

df_region = df_region.replace('Grand Est', 'Grand-Est')

df_region = df_region.replace('Île-de-France', 'Ile-de-France')

df_region = df_region.replace('Provence-Alpes-Côte d’Azur', "Provence-Alpes-Côte d'Azur")

df_region['maille_nom'].unique()

gb_region = df_region.groupby(['maille_nom']).max().reset_index() # get sum of cases by district

gb_region.rename(columns={'maille_nom':'region'},inplace=True) 

gb_region = gb_region.sort_values(by=['cases'], ascending=False) # sort descending



gb_region[:2]
plt.figure(figsize=(12, 12))



plt.subplot(311)

plt.bar(gb_region.region,gb_region['deaths'], color = 'k') # deaths by region

plt.ylabel('Current total deaths')

plt.xticks(rotation=90)

plt.xticks(gb_region['region']," ")

plt.title('Deaths by region', fontsize = 14)



plt.subplot(312)

plt.bar(gb_region.region,gb_region.hospitalises, color = 'tomato') #in hospital by region

plt.title('Total people hospitalized so far', fontsize = 14)

plt.ylabel('Hospitalized so far')

plt.xticks(gb_region['region']," ")



plt.subplot(313)

plt.bar(gb_region.region,gb_region['cases'], color = 'b') # cases by region

plt.ylabel('cases')

plt.xticks(rotation=90)

plt.title('Cases by region', fontsize = 14)



plt.suptitle('Overall regional stats', fontsize = 20)

plt.show()
reg_features = ['date','cases','deaths','reanimation','hospitalises','gueris'] # list relevant features
population13 = 1966005 

population75 = 2187526
# data for Marseille and Paris

df_13 = raw_data[raw_data.maille_code == 'DEP-13'] 

df_13 = df_13[reg_features] # select relevant features

df_75 = raw_data[raw_data.maille_code == 'DEP-75'] 

df_75 = df_75[reg_features] # select relevant features

df_13.tail()
plt.figure(figsize=(6,12))

plt.subplot(211)

plt.plot(df_75.date,(df_75['reanimation']/population75)*100, color = 'k', label =  'Paris')

plt.plot(df_13.date,(df_13['reanimation']/population13)*100, color = 'r', label =  'Bouches de Rhone') 

plt.ylabel('% population intensive care')

plt.xlim('2020-03-02', max(df_13.date))

plt.legend()

plt.title('% Population intensive care over time', fontsize = 14)



plt.subplot(212)

plt.plot(df_75.date,(df_75['hospitalises']/population75)*100, color = 'k', label =  'Paris')

plt.plot(df_13.date,(df_13['hospitalises']/population13)*100, color = 'r', label =  'Bouches de Rhone')

plt.ylabel('% population hospitalized')

plt.xlim('2020-03-02', max(df_13.date)) 

plt.xticks(rotation=90)

plt.legend()

plt.title('% Population hospitalized over time', fontsize = 14)



plt.suptitle('Paris et Bouche des Rhone districts', fontsize = 20)

plt.show()
# data for Ile de France (region around Paris)



df_IDF = raw_data[raw_data.maille_code == 'REG-11']

#df_IDF = df_IDF.dropna(subset=['cases']) # drop empty cells

df_IDF = df_IDF[reg_features] # select relevant features
# data for Provence (region around Marseille)

df_PACA = raw_data[raw_data.maille_code == 'REG-93'] 

#df_PACA = df_PACA.dropna(subset=['cases'])# drop empty cells

df_PACA = df_PACA[reg_features] # select relevant features
IDFpopulation = 12174880

PACApopulation = 5030890
plt.figure(figsize=(6, 12))



plt.subplot(211)

plt.plot(df_IDF.date,df_IDF['cases'], color = 'k', label =  'Ile de France') 

plt.plot(df_PACA.date,df_PACA['cases'], color = 'r', label =  'PACA') 

#plt.xlim('2020-03-15', max(df_PACA.date))   # set the xlim to left, right

plt.ylabel('COVID-19 cases')



plt.legend()

plt.title('Cases over time', fontsize = 14)



plt.subplot(212)

plt.plot(df_IDF.date,(df_IDF['hospitalises']/IDFpopulation)*100, color = 'k', label =  'Ile de France') # % in hospital of total Ile de France population

plt.plot(df_PACA.date,(df_PACA['hospitalises']/PACApopulation)*100, color = 'r', label =  'PACA') # % in hospital of total PACA population

plt.ylabel('% population hospitalized')

plt.xlim('2020-03-22', max(df_PACA.date)) 

plt.xticks(rotation=90)

plt.legend()

plt.title('% Population hospitalized over time', fontsize = 14)



plt.suptitle('Ile de France et PACA', fontsize = 20)

plt.show()
plt.figure(figsize=(6, 12))



plt.subplot(211)

plt.plot(df_IDF.date,(df_IDF['gueris']/IDFpopulation)*100, color = 'k', label =  'Ile de France') # % recovered of total Ile de France population

plt.plot(df_PACA.date,(df_PACA['gueris']/PACApopulation)*100, color = 'r', label =  'PACA') # % recovered of total PACA population

plt.ylabel('% population recovered from COVID-19')

plt.xlim('2020-03-22', max(df_PACA.date)) 

plt.legend()

plt.title('% Population recovered over time', fontsize = 14)



plt.subplot(212)

plt.plot(df_IDF.date,(df_IDF['deaths']/IDFpopulation)*100, color = 'k', label =  'Ile de France') # % deaths of total Ile de France population

plt.plot(df_PACA.date,(df_PACA['deaths']/PACApopulation)*100, color = 'r', label =  'PACA') # % deaths of total PACA population

plt.ylabel('% population deceased from COVID-19')

plt.xlim('2020-03-22', max(df_PACA.date)) 

plt.xticks(rotation=90)

plt.title('% Population dead over time', fontsize = 14)



plt.suptitle('Ile de France et PACA', fontsize = 20)

plt.show()
plt.figure(figsize=(18,6))

plt.plot(df_korea_trend['date'], df_korea_trend['cold'], label = 'Cold')

plt.plot(df_korea_trend['date'], df_korea_trend['flu'], label ='Flu')

plt.plot(df_korea_trend['date'], df_korea_trend['pneumonia'], label = 'Pneumonia')

plt.plot(df_korea_trend['date'], df_korea_trend['coronavirus'], label = 'Coronavirus')









plt.ylabel('%', fontsize=14)

plt.xlabel('Date', fontsize = 14)

plt.legend



plt.title('Korean diseases comparison', fontsize = 20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
df_korea.head()
df_korea_patient.head()
df_korea_patient.infection_case.value_counts()
# at what age can people get infected?

df_korea_age = df_korea_patient.groupby(['age']).count().reset_index()

df_korea_age = df_korea_age[['age','patient_id']]

plt.figure(figsize=(18,6))

plt.bar(df_korea_age['age'], df_korea_age['patient_id']) 

plt.xlabel('Age', fontsize=14)

plt.ylabel('Frequency', fontsize=14)



plt.suptitle('Age distribution', fontsize = 20)

plt.show()