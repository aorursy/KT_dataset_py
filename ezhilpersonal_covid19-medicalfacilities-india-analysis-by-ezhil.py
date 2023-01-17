# Created by Ezhilarasan Kannaiyan

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#df = pd.read_csv("G:\Python\Session6_26-Apr-2020\covid19-in-india\covid_19_india.csv") 

df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

gr1 = df.groupby('State/UnionTerritory') 

cured_death_comp = gr1.aggregate({'Cured':['sum'],'Deaths':['sum']}).sort_values(by=('Cured','sum'), ascending=False)

cured_death_comp.columns=['Tot_Cured','Total_Death']

#conf_death_comp.iloc[0:5,:].plot.bar()

st_names=list(cured_death_comp.iloc[:5,:].index)

cured_count = list(cured_death_comp.iloc[:5,0])

death_count = list(cured_death_comp.iloc[:5,1])

x=np.arange(5)

plt.bar(x+0.00,cured_count, width=0.25, label='Cured')

plt.bar(x+0.25,death_count, width=0.25,label='Death')

plt.xlabel('State Name')

plt.xticks(ticks=x,labels=st_names)

plt.ylabel('Count')

plt.title('Cured Vs Deaths Comparision')

plt.legend()

plt.grid(True)
#age_df = pd.read_csv("G:\Python\Session6_26-Apr-2020\Exercise\covid19-in-india\AgeGroupDetails.csv", index_col='Sno')

age_df = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv", index_col='Sno')
sorted_age = age_df.sort_values(by='Sno', ascending=False)

plt.barh(sorted_age.AgeGroup,sorted_age.TotalCases)

plt.xlabel('Total Cases')

plt.ylabel('Age Group')

plt.title('Total Cases by Age Group')

#Persons in Age Group 20 to 40 , got more affected due to travel/work (direct contact)
explode_capital = [0,0,0.1,0.1,0,0,0,0,0,0]  





plt.pie(age_df['TotalCases'],labels=age_df['AgeGroup'],explode=explode_capital, startangle=90, autopct='%1.1f%%', wedgeprops={'edgecolor':'black'})

plt.title('Total Cases by Age Group Pie Chart')

plt.show()
#bed_df = pd.read_csv("G:\Python\Session6_26-Apr-2020\Exercise\covid19-in-india\HospitalBedsIndia.csv", index_col='Sno')

bed_df = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv", index_col='Sno')



bed_df = bed_df[:-1]

bed_df.tail()
plt.figure(figsize=(13,8))

plt.bar(bed_df['State/UT'],bed_df['TotalPublicHealthFacilities_HMIS'])

plt.xticks(rotation=90, fontsize=18)

plt.xlabel('State Name', fontsize=22)

plt.yticks(fontsize=18)

plt.ylabel('Total_Beds', fontsize=22)

plt.show()
bed_df['Total_Hospitals'] = bed_df['TotalPublicHealthFacilities_HMIS'] + bed_df['NumRuralHospitals_NHP18'] +bed_df['NumUrbanHospitals_NHP18']

bed_df['Total_Beds'] = bed_df['NumPublicBeds_HMIS'] + bed_df['NumRuralBeds_NHP18'] +bed_df['NumUrbanBeds_NHP18']

bed_sorted5 = bed_df.sort_values(by='Total_Beds', ascending=False).head()

a = np.arange(1,6)

plt.bar(a-0.25,bed_sorted5['TotalPublicHealthFacilities_HMIS'],label='Public Health Centers', width=0.25)

plt.bar(a,bed_sorted5['NumRuralHospitals_NHP18'],label='Rural Hospitals', width=0.25)

plt.bar(a+0.25,bed_sorted5['NumUrbanHospitals_NHP18'],label='Urban Hospitals', width=0.25)



plt.legend()

plt.xticks(ticks=a,labels=bed_sorted5['State/UT'],rotation='vertical')

plt.xlabel('State Name')

plt.ylabel('Total Hospitals')

plt.title('State Vs Hospitals')

	
bed_sorted5 = bed_df.sort_values(by='Total_Beds', ascending=False).head()

a = np.arange(1,6)

plt.plot(bed_sorted5['State/UT'],bed_sorted5['NumPublicBeds_HMIS'],'o-',label='Public Health Center Beds')

plt.plot(bed_sorted5['State/UT'],bed_sorted5['NumRuralBeds_NHP18'],'v--',label='Rural Hospital Beds')

plt.plot(bed_sorted5['State/UT'],bed_sorted5['NumUrbanBeds_NHP18'],'+-',label='Urban Hospital Beds')



plt.legend()

plt.xticks(rotation='vertical')

plt.xlabel('State Name')

plt.ylabel('Total Beds')

plt.title('State Vs Beds')

#Hospital and Bed Facilities in India (Top 5 states)