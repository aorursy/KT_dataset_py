import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
#import time#for enhancin the image of graph

#import psutil
#%matplotlib notebook

#plt.rcParams['animation.html'] = 'jshtml'#java script & html structure

data = pd.read_csv("../input/covid19-dataset-of-india/covid_19_india.csv")
data.describe()
data.info()
data.tail()
data.columns
no_use_rows = data.loc[pd.to_numeric(data['ConfirmedIndianNational'], errors='coerce').isnull()]#finding strings in columns

no_use_rows
del data['ConfirmedIndianNational']#deleting column 

data.head()
del data['ConfirmedForeignNational']#deleting column 
data.head(2119)
data.tail(50)
data.rename(columns = {'State/UnionTerritory':'StatesUniounTerritories'}, inplace = True)

data
data["StatesUniounTerritories"]= data["StatesUniounTerritories"].str.replace(r'\W',"")#removing all the special characters

#using (r'\W',"") also it replaces the string with other name if we want

data
data["StatesUniounTerritories"]= data["StatesUniounTerritories"].str.replace('Telengana', "Telangana")#replacing string

#telengana to telangana

data
data["StatesUniounTerritories"]= data["StatesUniounTerritories"].str.replace('DadarNagarHaveli', "DadraandNagarHaveliandDamanandDiu")#replacing string

#DadarNagarHaveli to DadraandNagarHaveliandDamanandDiu

data
data["StatesUniounTerritories"]= data["StatesUniounTerritories"].str.replace('DamanDiu', "DadraandNagarHaveliandDamanandDiu")#replacing string

#DamanDiu to DadraandNagarHaveliandDamanandDiu

data
cured = data.groupby('StatesUniounTerritories').Cured.max()#seeing maximum cured by grouping satates 

cured
cured.plot(kind='bar', figsize=(14, 6),color='g')

plt.ylabel('No. of patients recovered (in lakhs)')

plt.xlabel('States & Union Territories')

plt.title('Covid19 recovered patients data in States & Union Territories of india')

plt.savefig('cured patients data in bar graph.png', bbox_inches = 'tight')

plt.grid()

plt.show()
max_deaths = data.groupby('StatesUniounTerritories').Deaths.max()#seeing maximum death by grouping satates

max_deaths
max_deaths.plot(kind='bar', figsize=(14, 6),color='red')

plt.ylabel('Number of deaths')

plt.xlabel('States & Union Territories')

plt.title('Covid19 data on number of deaths in States & Union Territories of india')

plt.grid()

plt.savefig('death cases data bar graph2.png', bbox_inches = 'tight')
max_cases = data.groupby('StatesUniounTerritories').Confirmed.max()#seeing maximum cases in states by grouping satates

max_cases
max_cases.sum()
max_cases.plot(kind='bar', figsize=(14, 6))

plt.ylabel('Cases registered (in lakhs)')

plt.xlabel('States & Union Territories')

plt.grid()

plt.title('Covid19 cases registered in States & Union Territories of India')

plt.savefig("cases registered in states2.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size
#how many states affected

data['StatesUniounTerritories'].unique()
len(data['StatesUniounTerritories'].unique())#how many states affected
### Adding months column 
data['Month'] = data['Date'].str[3:5]

data['Month'] = data['Month'].astype('int64')

data.head(5)



data.groupby(['Month']).sum()#determining step1 for groupby to choosing particuler column
results = data.groupby(['Month'])[["Confirmed", "Cured", "Deaths"]].max()#selecting particular columns to do groupby result

results    
data.dtypes#checking formats
data['Time']= pd.to_datetime(data.Time)
data.tail(50)
del data['Time']#deleting column 
data.head()
all_cured = cured.value_counts

all_cured
all_cases = max_cases.value_counts()

all_cases
all_deaths = max_deaths.value_counts()

all_deaths
all_states = data.StatesUniounTerritories#seeing only particuler column

all_states
telangana_data = data[data.StatesUniounTerritories.str.contains('Telangana')]#seeing only telengana values for finding incorrect spelling

telangana_data
month_telangana = telangana_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_telangana
In [11]: month_telangana
month_telangana.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases(in lakhs)')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Telangana State')

plt.savefig("cases registered in telangana.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

telangana_data.max()
AndamanandNicobarIslands_data = data[data.StatesUniounTerritories.str.contains('AndamanandNicobarIslands')]#seeing only telengana values for finding incorrect spelling

AndamanandNicobarIslands_data
month_AndamanandNicobarIslands = AndamanandNicobarIslands_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_AndamanandNicobarIslands
month_AndamanandNicobarIslands.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Andaman and Nicobar Islands')

plt.savefig("cases registered in AndamanandNicobarIslands.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

AndamanandNicobarIslands_data.max()
AndhraPradesh_data = data[data.StatesUniounTerritories.str.contains('AndhraPradesh')]

AndhraPradesh_data
month_AndhraPradesh = AndhraPradesh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_AndhraPradesh
month_AndhraPradesh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Andhra Pradesh')

plt.savefig("cases registered in AndhraPradesh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

AndhraPradesh_data.max()
ArunachalPradesh_data = data[data.StatesUniounTerritories.str.contains('ArunachalPradesh')]

ArunachalPradesh_data
month_ArunachalPradesh = ArunachalPradesh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_ArunachalPradesh
month_ArunachalPradesh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Arunachal Pradesh')

plt.savefig("cases registered in ArunachalPradesh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

ArunachalPradesh_data.max()
Assam_data = data[data.StatesUniounTerritories.str.contains('Assam')]

Assam_data
month_Assam = Assam_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Assam
month_Assam.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Assam')

plt.savefig("cases registered in Assam.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Assam_data.max()
Bihar_data = data[data.StatesUniounTerritories.str.contains('Bihar')]

Bihar_data
month_Bihar = Bihar_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Bihar
month_Bihar.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Bihar')

plt.savefig("cases registered in Bihar.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Chandigarh_data = data[data.StatesUniounTerritories.str.contains('Chandigarh')]

Chandigarh_data
month_Chandigarh = Chandigarh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Chandigarh
month_Chandigarh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Chandigarh')

plt.savefig("cases registered in Chandigarh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Chhattisgarh_data = data[data.StatesUniounTerritories.str.contains('Chhattisgarh')]

Chandigarh_data
month_Chhattisgarh = Chhattisgarh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Chhattisgarh
month_Chhattisgarh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Chhattisgarh')

plt.savefig("cases registered in Chhattisgarh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

DadraandNagarHaveliandDamanandDiu_data = data[data.StatesUniounTerritories.str.contains('DadraandNagarHaveliandDamanandDiu')]

DadraandNagarHaveliandDamanandDiu_data
month_DadraandNagarHaveliandDamanandDiu = DadraandNagarHaveliandDamanandDiu_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_DadraandNagarHaveliandDamanandDiu
month_DadraandNagarHaveliandDamanandDiu.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in DadraandNagarHaveliandDamanandDiu')

plt.savefig("cases registered in DadraandNagarHaveliandDamanandDiu.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Delhi_data = data[data.StatesUniounTerritories.str.contains('Delhi')]

Delhi_data
month_Delhi = Delhi_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Delhi
month_Delhi.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Delhi')

plt.savefig("cases registered in Delhi.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size
Gujarat_data = data[data.StatesUniounTerritories.str.contains('Gujarat')]

Gujarat_data
month_Gujarat = Gujarat_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Gujarat
month_Gujarat.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Gujarat')

plt.savefig("cases registered in Gujarat.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size
Goa_data = data[data.StatesUniounTerritories.str.contains('Goa')]

Goa_data
month_Goa = Goa_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Goa
month_Goa.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Goa')

plt.savefig("cases registered in Goa.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Haryana_data = data[data.StatesUniounTerritories.str.contains('Haryana')]

Haryana_data
month_Haryana = Haryana_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Haryana
month_Haryana.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Haryana')

plt.savefig("cases registered in Haryana.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

HimachalPradesh_data = data[data.StatesUniounTerritories.str.contains('HimachalPradesh')]

HimachalPradesh_data
month_HimachalPradesh = HimachalPradesh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_HimachalPradesh
month_HimachalPradesh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in HimachalPradesh')

plt.savefig("cases registered in HimachalPradesh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

JammuandKashmir_data = data[data.StatesUniounTerritories.str.contains('JammuandKashmir')]

JammuandKashmir_data
month_JammuandKashmir = JammuandKashmir_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_JammuandKashmir
month_JammuandKashmir.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in JammuandKashmir')

plt.savefig("cases registered in JammuandKashmir.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Jharkhand_data = data[data.StatesUniounTerritories.str.contains('Jharkhand')]

Jharkhand_data
month_Jharkhand = Jharkhand_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Jharkhand
month_Jharkhand.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Jharkhand')

plt.savefig("cases registered in Jharkhand.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Karnataka_data = data[data.StatesUniounTerritories.str.contains('Karnataka')]

Karnataka_data
month_Karnataka = Karnataka_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Karnataka
month_Karnataka.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Karnataka')

plt.savefig("cases registered in Karnataka.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Kerala_data = data[data.StatesUniounTerritories.str.contains('Kerala')]

Kerala_data
month_Kerala = Kerala_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Kerala
month_Kerala.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Kerala')

plt.savefig("cases registered in Kerala.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Ladakh_data = data[data.StatesUniounTerritories.str.contains('Ladakh')]

Ladakh_data
month_Ladakh = Goa_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Ladakh
month_Ladakh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Ladakh')

plt.savefig("cases registered in Ladakh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

MadhyaPradesh_data = data[data.StatesUniounTerritories.str.contains('MadhyaPradesh')]

MadhyaPradesh_data
month_MadhyaPradesh = MadhyaPradesh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_MadhyaPradesh
month_MadhyaPradesh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in MadhyaPradesh')

plt.savefig("cases registered in Goa.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Maharashtra_data = data[data.StatesUniounTerritories.str.contains('Maharashtra')]

Maharashtra_data
month_Maharashtra = Maharashtra_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Maharashtra
month_Maharashtra.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases(in lakhs)')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Maharashtra')

plt.savefig("cases registered in Maharashtra.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Manipur_data = data[data.StatesUniounTerritories.str.contains('Manipur')]

Manipur_data
month_Manipur = Goa_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Manipur
month_Manipur.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Manipur')

plt.savefig("cases registered in Manipur.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Meghalaya_data = data[data.StatesUniounTerritories.str.contains('Meghalaya')]

Meghalaya_data
month_Meghalaya = Manipur_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Meghalaya
month_Meghalaya.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Meghalaya')

plt.savefig("cases registered in Meghalaya.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Mizoram_data = data[data.StatesUniounTerritories.str.contains('Mizoram')]

Mizoram_data
month_Mizoram = Mizoram_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Mizoram
month_Mizoram.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Mizoram')

plt.savefig("cases registered in Mizoram.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Nagaland_data = data[data.StatesUniounTerritories.str.contains('Goa')]

Nagaland_data
month_Nagaland = Nagaland_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Nagaland
month_Nagaland.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Nagaland')

plt.savefig("cases registered in Nagaland.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Goa_data = data[data.StatesUniounTerritories.str.contains('Goa')]

Goa_data
month_Goa = Goa_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Goa
month_Meghalaya.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Meghalaya')

plt.savefig("cases registered in Meghalaya.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Odisha_data = data[data.StatesUniounTerritories.str.contains('Odisha')]

Odisha_data
month_Odisha = Odisha_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Odisha
month_Odisha.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Odisha')

plt.savefig("cases registered in Odisha.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Puducherry_data = data[data.StatesUniounTerritories.str.contains('Puducherry')]

Puducherry_data
month_Puducherry = Puducherry_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Puducherry
month_Puducherry.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Puducherry')

plt.savefig("cases registered in Puducherry.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Punjab_data = data[data.StatesUniounTerritories.str.contains('Punjab')]

Punjab_data
month_Punjab = Punjab_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Punjab
month_Punjab.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Punjab')

plt.savefig("cases registered in Punjab.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Rajasthan_data = data[data.StatesUniounTerritories.str.contains('Rajasthan')]

Rajasthan_data
month_Rajasthan = Rajasthan_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Rajasthan
month_Rajasthan.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Rajasthan')

plt.savefig("cases registered in Rajasthan.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Sikkim_data = data[data.StatesUniounTerritories.str.contains('Sikkim')]

Sikkim_data
month_Sikkim = Sikkim_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Sikkim
month_Sikkim.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Sikkim')

plt.savefig("cases registered in Sikkim.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

TamilNadu_data = data[data.StatesUniounTerritories.str.contains('TamilNadu')]

TamilNadu_data
month_TamilNadu = TamilNadu_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_TamilNadu
month_TamilNadu.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in TamilNadu')

plt.savefig("cases registered in TamilNadu.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Tripura_data = data[data.StatesUniounTerritories.str.contains('Tripura')]

Tripura_data
month_Tripura = Tripura_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Tripura
month_Tripura.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Tripura')

plt.savefig("cases registered in Tripura.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

UttarPradesh_data = data[data.StatesUniounTerritories.str.contains('UttarPradesh')]

UttarPradesh_data
month_UttarPradesh = UttarPradesh_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_UttarPradesh
month_UttarPradesh.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in UttarPradesh')

plt.savefig("cases registered in UttarPradesh.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

Uttarakhand_data = data[data.StatesUniounTerritories.str.contains('Uttarakhand')]

Uttarakhand_data
month_Uttarakhand = Uttarakhand_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_Uttarakhand
month_Uttarakhand.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in Uttarakhand')

plt.savefig("cases registered in Uttarakhand.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

WestBengal_data = data[data.StatesUniounTerritories.str.contains('WestBengal')]

WestBengal_data
month_WestBengal = WestBengal_data.groupby(['Month'])[['Confirmed', 'Cured', 'Deaths']].max()

month_WestBengal
month_WestBengal.plot(kind='bar', figsize=(16, 6))

plt.grid()

plt.ylabel('Covid19 cases')

plt.xlabel('Months')

plt.title('Covid19 cases registered in different months in WestBengal')

plt.savefig("cases registered in WestBengal.png", bbox_inches = 'tight')#to save this figure, filename.png, for fitin the size

results = data.groupby(['Month'])[["Confirmed", "Cured", "Deaths"]].max()#selecting particular columns to do groupby result
results
data.isnull().sum()#finding null values
data.columns