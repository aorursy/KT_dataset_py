
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


data_covid = pd.read_excel("../input/COVID-19-April-xl.xlsx")
data_covid
data_view = data_covid[['day','month','year','countryterritoryCode','cases','deaths']]
data_view.head().style.background_gradient(cmap='Greens')
#data_view

# Italy = ITA ,  Spain = ESP , France =  FRA  , United States of America  = USA ,  United Kingdom = GBR
data_country_PAK_March  = data_view.loc[(data_view['countryterritoryCode'] == "PAK") & (data_view['cases'] > 0)& (data_view['month'] == 3)]
data_country_PAK_April  = data_view.loc[(data_view['countryterritoryCode'] == "PAK") & (data_view['cases'] > 0)& (data_view['month'] == 4)]

#data_pak

plt.figure(figsize=(20,9))
plt.title('COVID-19  Patients in Pakistan (March & April )')
plt.xlabel('Day')
plt.ylabel('Number of Patients')

plt.plot(data_country_PAK_March.day  , data_country_PAK_March.cases , 'b.-',label ="Cases March")
plt.plot(data_country_PAK_March.day  , data_country_PAK_March.deaths , 'y.-', label="Deaths March")

plt.plot(data_country_PAK_April.day  , data_country_PAK_April.cases , 'g.-',label ="Cases April")
plt.plot(data_country_PAK_April.day  , data_country_PAK_April.deaths , 'r.-', label="Deaths April")

plt.legend()

data_country_PAK_April.head().style.background_gradient(cmap='Greens')

April_data = [[data_country_PAK_April.cases.sum()],[data_country_PAK_April.deaths.sum()]]
March_data = [[data_country_PAK_March.cases.sum()],[data_country_PAK_March.deaths.sum()]]

plt.title('Covid-19 Pak April Data')
my_labels = 'Cases','Deaths'
plt.pie(April_data,labels=my_labels,autopct='%1.1f%%')
plt.axis('equal')

plt.show()

March_data = [[data_country_PAK_March.cases.sum()],[data_country_PAK_March.deaths.sum()]]
plt.title('Covid-19 Pak March Data')
plt.pie(March_data,labels=my_labels,autopct='%1.1f%%')
#plt.axis('equal')

plt.show()