# 1.2 Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%reset -f
# 1.0 For data manipulation
import numpy as np
import pandas as pd
# 1.1 For plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
testdetail = pd.read_csv("../input/covid19-in-india/ICMRTestingDetails.csv",parse_dates=['DateTime'],dayfirst=True)
testdetail.drop(testdetail[testdetail['TotalIndividualsTested'].isna() & testdetail['TotalPositiveCases'].isna()].index,inplace=True)
testdetail.loc[testdetail['TotalIndividualsTested'].isna()==True,'TotalIndividualsTested']=testdetail.loc[testdetail['TotalIndividualsTested'].isna()==True,'TotalSamplesTested']
testdetail.drop(testdetail[testdetail['TotalPositiveCases'].isna()].index,inplace=True)
testdetail['PositivePercentage'] = testdetail['TotalPositiveCases']/testdetail['TotalIndividualsTested']*100
testdetail['Date'] = testdetail['DateTime'].dt.date.apply(lambda x: x.strftime('%Y-%m-%d'))
testdetail["lockdown"] = pd.cut(
                       testdetail['DateTime'],
                       bins= [pd.datetime(year=2020, month=3, day=1),pd.datetime(year=2020, month=3, day=25),pd.datetime(year=2020, month=4, day=14),
                              pd.datetime(year=2020, month=5, day=3)],
                       labels= ["No-Lockdown", "Lockdown-1", "Lockdown-2"]
                      )

fig, ax = plt.subplots(figsize=(10,10)) 
sns.barplot(x = 'Date',
            y = 'TotalIndividualsTested',
            hue = 'lockdown',   
            ci = 95,
            data =testdetail)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
fig, ax = plt.subplots(figsize=(10,10)) 
sns.barplot(x = 'Date',
            y = 'PositivePercentage',
            hue = 'lockdown',   
            ci = 95,
            data =testdetail,
            ax = ax)
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, ha='right')
plt.plot('TotalIndividualsTested','PositivePercentage','bo-',data = testdetail)
plt.title("Line Plot for Testing vs PositivePercentage")
plt.xlabel("TotalIndividualsTested")
plt.ylabel("PositivePercentage")
#Data Cleansing
Stwisetestdetail = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv",parse_dates=['Date'])
Stwisetestdetail.drop(Stwisetestdetail[Stwisetestdetail['Date']=='2020-02-16'].index,inplace=True)
Stwisetestdetail.drop(Stwisetestdetail[Stwisetestdetail['Negative'].isna() & Stwisetestdetail['Positive'].isna()].index,inplace=True)
Stwisetestdetail.loc[Stwisetestdetail['Negative'].isna(),'Negative'] = Stwisetestdetail.loc[Stwisetestdetail['Negative'].isna(),'TotalSamples']-Stwisetestdetail.loc[Stwisetestdetail['Negative'].isna(),'Positive']
Stwisetestdetail.loc[Stwisetestdetail['Positive'].isna(),'Positive'] = Stwisetestdetail.loc[Stwisetestdetail['Positive'].isna(),'TotalSamples']-Stwisetestdetail.loc[Stwisetestdetail['Positive'].isna(),'Negative']
stateDategrouped = Stwisetestdetail.groupby([Stwisetestdetail['Date'].dt.date, 'State'])
stateDateStatus = stateDategrouped[['TotalSamples','Negative','Positive']].sum()
stateDateStatus = stateDateStatus.reset_index()
stateDateStatus['Date'] = pd.to_datetime(stateDateStatus['Date'])
stateDateStatus['TestingMonth'] = stateDateStatus['Date'].dt.month.map({
                                    4: 'April',
                                    5: 'May'
                                    }
                                )
stateDateStatus['TestingDay'] = stateDateStatus['Date'].dt.day
stateDateStatus['PositivePercent'] = (stateDateStatus['Positive']/stateDateStatus['TotalSamples'])*100
stateDateStatus['Date'] = stateDateStatus['Date'].dt.date

#pivot State data Datewise 
state_date = stateDateStatus.pivot(index='Date', columns='State', values='PositivePercent')
fig, ax = plt.subplots(figsize=(8,8)) 
sns.heatmap(state_date.fillna(0),cmap="YlOrRd",xticklabels=True,ax=ax)
ax.invert_yaxis()
sns.catplot(x = 'TestingDay',
            y = 'PositivePercent',
            row = 'State',
            col = 'TestingMonth',
            kind = 'bar',
            data =stateDateStatus)
#sns.relplot(x='TestingMonth',y='PositivePercent',row = 'State',kind = 'scatter',data=stateDateStatus)
#Loading Population data for States
census = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
census['AreaInNumbers'] = census['Area'].apply(lambda x : x.split("km2")[0].replace(",", ""))
census['DensityInNumbers'] = census['Density'].apply(lambda x : x.split("/km2")[0].replace(",", ""))
census['AreaInNumbers']=census['AreaInNumbers'].astype('int32')
census['DensityInNumbers']=census['DensityInNumbers'].astype('float')
#Latest Covid -19 status of Indian States
stateWiseCovid = Stwisetestdetail.groupby('State')[['TotalSamples','Negative','Positive']].max()
#Joining State census data with Covid-19 data of States
stateCensusCovid = pd.merge(census,stateWiseCovid,left_on = 'State / Union Territory',right_on = 'State', how='left')[['State / Union Territory','Population','Rural population','Urban population','Gender Ratio', 'AreaInNumbers',
       'DensityInNumbers','TotalSamples', 'Negative', 'Positive']]
stateCensusCovid = stateCensusCovid.fillna(0)
stateCensusCovid['PositiveinPercent'] = stateCensusCovid['Positive']/stateCensusCovid['TotalSamples']*100
stateCensusCovid['UrbanPopulationInPercent'] = stateCensusCovid['Urban population']/stateCensusCovid['Population']*100
stateCensusCovid['RuralPopulationInPercent'] = 100 -stateCensusCovid['UrbanPopulationInPercent']
#Finding Relation between Density of Population and no. of positive cases.
sns.jointplot(x=stateCensusCovid.DensityInNumbers,y='Positive',xlim=(0,2000),ylim=(0,5000),data = stateCensusCovid.fillna(0))

#sns.jointplot(x=stateCensusCovid['Urban population']/stateCensusCovid['Population'],y='Positive',data = stateCensusCovid.fillna(0),kind="kde")
sns.jointplot(x='UrbanPopulationInPercent',y='PositiveinPercent',xlim=(0,100),data = stateCensusCovid,kind="kde")
sns.jointplot(x='DensityInNumbers',y='PositiveinPercent',xlim=(0,1500),data = stateCensusCovid,kind="hex")
#Loading State health-Infrastructure data.
statehealthInfra = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
statehealthInfra['TotalHospitals'] = statehealthInfra['NumRuralHospitals_NHP18'] + statehealthInfra['NumUrbanHospitals_NHP18']
statehealthInfra['TotalBeds'] = statehealthInfra['NumRuralBeds_NHP18'] + statehealthInfra['NumUrbanBeds_NHP18']
#Merging Health-Infrastructure data with census and Covid-19 cases
stateCensusCovid = pd.merge(stateCensusCovid,statehealthInfra,left_on = 'State / Union Territory',right_on = 'State/UT', how='left')[['State / Union Territory', 'Population', 'Rural population',
       'Urban population', 'Gender Ratio', 'AreaInNumbers', 'DensityInNumbers',
       'TotalSamples', 'Negative', 'Positive', 'PositiveinPercent',
       'UrbanPopulationInPercent', 'RuralPopulationInPercent','TotalPublicHealthFacilities_HMIS','NumPublicBeds_HMIS', 'NumRuralHospitals_NHP18', 'NumRuralBeds_NHP18',
       'NumUrbanHospitals_NHP18', 'NumUrbanBeds_NHP18', 'TotalHospitals',
       'TotalBeds']]
stateCensusCovid['RuralPersonPerBed'] = stateCensusCovid['Rural population']/stateCensusCovid['NumRuralBeds_NHP18']
stateCensusCovid['UrbanPersonPerBed'] = stateCensusCovid['Urban population']/stateCensusCovid['NumUrbanBeds_NHP18']
stateCensusCovid['PersonPerBed'] = stateCensusCovid['Population']/stateCensusCovid['TotalBeds']
stateCensusCovid['PersonPerPublicBed'] = stateCensusCovid['Population']/stateCensusCovid['NumPublicBeds_HMIS']
#sns.jointplot(x='Positive',y=stateCensusCovid['Population']/stateCensusCovid['NumPublicBeds_HMIS'],xlim=(0,3000),ylim=(0,3000),data = stateCensusCovid,kind="kde")
fig, ax = plt.subplots(figsize=(10,10)) 
sns.barplot(x = 'PersonPerPublicBed',
            y = 'State / Union Territory',
           data =stateCensusCovid)
sns.jointplot(x='UrbanPersonPerBed',y='PositiveinPercent',data = stateCensusCovid,kind="kde")
sns.jointplot(x='TotalBeds',y='Positive',xlim=(0,30000),ylim=(0,10000),data = stateCensusCovid,kind="kde")
