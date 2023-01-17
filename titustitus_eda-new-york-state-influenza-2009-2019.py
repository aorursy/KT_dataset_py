import pandas as pd

import numpy as np

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import numpy as np

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('../input/h1n1-new-york-2009/Influenza_NY.csv')

df=df.drop(df.columns[0],axis='columns') #first column is useless

df['Prob_infected']=df['Infected']/df['Population'] # Probability of being infected (Weekly)

df['Density']=df['Population']/df['Area'] # Density of population of the area

df['Week Ending Date']=pd.to_datetime(df['Week Ending Date'], format='%m/%d/%Y')
df_p=df.groupby(['County','Season'])['Prob_infected'].sum().reset_index()

df_p=df_p.groupby(['County']).mean()['Prob_infected'].reset_index()

plt.figure(figsize=(16, 6))

plt.title('Probability of influenza positive test per County during a Season (Oct-May)',fontsize=15)

sns.set()

ax=sns.barplot(x='County',y='Prob_infected',data=df_p)

for item in ax.get_xticklabels():

    item.set_rotation(90)

df_p=df.groupby(['County','Week Ending Date'])['Prob_infected'].sum().reset_index()

plt.figure(figsize=(16, 6))

plt.title('Probability of influenza positive test per County',fontsize=15)

sns.set()

ax = sns.lineplot(x='Week Ending Date', y="Prob_infected", hue="County",data=df_p[['Week Ending Date','Prob_infected','County']],lw=0.6,legend=False)
df_p=df.groupby(['County','Month'])['Prob_infected'].sum().reset_index()

plt.figure(figsize=(16, 6))

plt.title('Monthly probability of influenza positive test per County',fontsize=15)

sns.set()

df_p=df.groupby(['County','Year','Month'])['Prob_infected'].sum().reset_index()

df_p=df_p.groupby(['County','Month']).mean()['Prob_infected'].reset_index() # Mean per month 



ax = sns.lineplot(x='Month', y="Prob_infected", hue="County",data=df_p[['Month','Prob_infected','County']],marker='o',lw=0.8,legend=False)



#sns.set()
df_p=df.groupby(['County','Year','Month'])['Prob_infected'].sum().reset_index()

df_p=df_p.groupby(['County','Month']).mean()['Prob_infected'].reset_index() # Mean per month 

Most_infected=df_p.groupby(['County']).sum()['Prob_infected'].reset_index().sort_values('Prob_infected').iloc[-15::]['County']

Least_infected=df_p.groupby(['County']).sum()['Prob_infected'].reset_index().sort_values('Prob_infected').iloc[0:15]['County']

plt.figure(figsize=(16, 6))

plt.title('Monthly probability of influenza positive test per County',fontsize=15)

sns.set()

selected_counties=np.concatenate([Most_infected[-4::],Least_infected[0:4]])



df_p=df_p[df_p['County'].isin(selected_counties)]



ax = sns.lineplot(x='Month', y="Prob_infected", hue="County",data=df_p[['Month','Prob_infected','County']],marker='o',lw=1,legend='full')

df_p=df[['County','Year','Population','Beds_hospital','Service_hospital']]

df_p=df_p[df_p['Year']==2017]

df_p['Prop_hosp_beds']=df_p['Beds_hospital']/df_p['Population']

df_p['Prop_hosp_service']=df_p['Service_hospital']/df_p['Population']

df_p=df_p.drop_duplicates()

plt.figure(figsize=(16, 6))

plt.title('2020 proportion of hospital beds per County',fontsize=15)



sns.set()

ax=sns.barplot(x='County',y='Prop_hosp_beds',data=df_p)



for item in ax.get_xticklabels():

    item.set_rotation(90)

sns.set()



plt.figure(figsize=(16, 6))

plt.title('2020 proportion of hospital service per County',fontsize=15)

sns.set()

ax=sns.barplot(x='County',y='Prop_hosp_service',data=df_p)

for item in ax.get_xticklabels():

    item.set_rotation(90)

sns.set()





df_p=df[['County','Year','Population','Discharges_Other_Hospital_intervention',

       'Discharges_Respiratory_system_interventions',

       'Total_Charge_Other_Hospital_intervention',

       'Total_Charge_Respiratory_system_interventions']]

df_p=df_p.groupby(['County','Year','Population']).sum().reset_index()

df_p=df_p[df_p['Year'].isin([2009,2010,2011,2012,2013,2014,2015,2016,2017])]

df_p=df_p.groupby(['County','Population']).mean().reset_index()



df_p['Discharges_respiratory_pp']=df_p['Discharges_Respiratory_system_interventions']/df_p['Population']

df_p.head()

plt.figure(figsize=(16, 6))

plt.title('Average respiratory medical interventions per person',fontsize=15)

sns.set()

ax=sns.barplot(x='County',y='Discharges_respiratory_pp',ci=None,data=df_p)

for item in ax.get_xticklabels():

    item.set_rotation(90)



df_p=df[['County','Year','Population','Discharges_Other_Hospital_intervention',

       'Discharges_Respiratory_system_interventions',

       'Total_Charge_Other_Hospital_intervention',

       'Total_Charge_Respiratory_system_interventions']]

df_p=df_p.groupby(['County','Year','Population']).sum().reset_index()

df_p=df_p[df_p['Year'].isin([2009,2010,2011,2012,2013,2014,2015,2016,2017])]

df_p=df_p.groupby(['County','Population']).mean().reset_index()

del df_p['Year']



df_p['Totcharge_respiratory_pp']=df_p['Total_Charge_Respiratory_system_interventions']/df_p['Population']

df_p.head()

plt.figure(figsize=(16, 6))

plt.title('Average total expenditure of respiratory medical interventions per person',fontsize=15)

sns.set()

ax=sns.barplot(x='County',y='Totcharge_respiratory_pp',ci=None,data=df_p)

for item in ax.get_xticklabels():

    item.set_rotation(90)

sns.set()

#df_p=df_p[df_p['County'].isin(Most_infected)]

#df_p.head(10)
df_p=df.groupby(['County','Year','Month'])['Prob_infected'].sum().reset_index()

df_p=df_p.rename(columns={'Prob_infected':'Prob_month_infected'})

#df_p=df_p.groupby(['County','Year''Month']).mean()['Prob_infected'].reset_index() # Mean per month 



df=df.set_index(['County','Year','Month']).join(df_p.set_index(['County','Year','Month']))

df=df.reset_index()



df_p=df.groupby(['County','Season'])['Prob_infected'].sum().reset_index()

df_p=df_p.rename(columns={'Prob_infected':'Prob_season_infected'})





df=df.set_index(['County','Season']).join(df_p.set_index(['County','Season']))

df=df.reset_index()
df_p=df.groupby(['County','Year','Population','Month','Season','Region']).mean().reset_index()

df_month=df_p[['County', 'Year', 'Month','Avg household size','Area',

       'Population', 'Under_18', '18-24', '25-44', '45-64', 'Above_65',

       'Median_age', 'Medianfamilyincome', 'Number_households',

       'Beds_adult_facility_care', 'Beds_hospital', 'County_Served_hospital',

       'Service_hospital', 'Discharges_Other_Hospital_intervention',

       'Discharges_Respiratory_system_interventions',

       'Total_Charge_Other_Hospital_intervention',

       'Total_Charge_Respiratory_system_interventions', 'Unemp_rate', 'Density', 'Prob_month_infected','Prob_season_infected']]

df_p=df.groupby(['County','Season','Region']).mean().reset_index()

df_season=df_p[['County','Season','Avg household size','Area',

       'Population', 'Under_18', '18-24', '25-44', '45-64', 'Above_65',

       'Median_age', 'Medianfamilyincome', 'Number_households',

       'Beds_adult_facility_care', 'Beds_hospital', 'County_Served_hospital',

       'Service_hospital', 'Discharges_Other_Hospital_intervention',

       'Discharges_Respiratory_system_interventions',

       'Total_Charge_Other_Hospital_intervention',

       'Total_Charge_Respiratory_system_interventions', 'Unemp_rate', 'Density','Prob_season_infected']]
df_season=df_season.dropna()

all_columns = "+".join(df_season.columns)

all_columns=all_columns.replace('County+','')

all_columns=all_columns.replace('+Prob_season_infected','')

my_formula = "Prob_season_infected ~" + all_columns

import statsmodels.api as sm

import statsmodels.formula.api as smf

md=smf.mixedlm("Prob_season_infected ~Area+Population+Under_18+Above_65+Median_age+Medianfamilyincome+Number_households+Beds_adult_facility_care+Beds_hospital+County_Served_hospital+Service_hospital+Discharges_Other_Hospital_intervention+Discharges_Respiratory_system_interventions+Total_Charge_Other_Hospital_intervention+Total_Charge_Respiratory_system_interventions+Unemp_rate+Density", df_season,groups=df_season["County"])

mdf = md.fit()

print(mdf.summary())