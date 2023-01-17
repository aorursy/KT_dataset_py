# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/covid-19-dataset/covid_19_clean_complete.csv")
data.head()
index_for_na_val = data['Province/State'].loc[data['Province/State'].isna()].index
data['Province/State'].iloc[index_for_na_val] = data['Country/Region'].iloc[index_for_na_val]
sns.heatmap(data.isna())
data.info()
data.sort_values(by = ['Date'], inplace = True)
star_date = data['Date'].unique()[0]

end_date = data['Date'].unique()[-1]



print(f"We have data from {star_date} to {end_date}")
data.columns
len(data['Confirmed'].loc[data['Country/Region'] == 'China'].sort_values().loc[data['Confirmed'] < 10])
country_wise_data = data.groupby('Country/Region').max().reset_index()
len(country_wise_data['Country/Region'])
cwd = country_wise_data.loc[country_wise_data['Confirmed'] > 1000]

plt.figure(figsize = (16, 16))

plt.subplot(3, 3, 1)

plt.bar(cwd['Country/Region'], cwd['Confirmed'])

plt.ylabel('No of cases till 09/03/2020')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 2)

plt.bar(cwd['Country/Region'], cwd['Deaths']/cwd['Confirmed'])

plt.ylabel('Death percentage till 09/03/2020')

plt.title('No of conformed cases above 1000 by countries')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 3)

plt.bar(cwd['Country/Region'], cwd['Recovered']/cwd['Confirmed'])

plt.ylabel('Death percentage till 09/03/2020')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 4)

cwd = country_wise_data.loc[(country_wise_data['Confirmed'] > 100) & (country_wise_data['Confirmed'] < 1000)]

plt.bar(cwd['Country/Region'], cwd['Confirmed'])

plt.ylabel('No of cases till 09/03/2020')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 5)

plt.bar(cwd['Country/Region'], cwd['Deaths']/cwd['Confirmed'])

plt.ylabel('Death percentage till 09/03/2020')

plt.title('No of conformed cases between 100 to 1000 by countries')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 6)

plt.bar(cwd['Country/Region'], cwd['Recovered']/cwd['Confirmed'])

plt.ylabel('Death percentage till 09/03/2020')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 7)

cwd = country_wise_data.loc[(country_wise_data['Confirmed'] > 50) & (country_wise_data['Confirmed'] < 100)]

plt.bar(cwd['Country/Region'], cwd['Confirmed'])

plt.xlabel('Country')

plt.ylabel('No of cases till 09/03/2020')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 8)

plt.bar(cwd['Country/Region'], cwd['Deaths']/cwd['Confirmed'])

plt.ylabel('Death percentage till 09/03/2020')

plt.title('No of conformed cases between 50 to 100 by countries')

plt.xticks(rotation = 90)



plt.subplot(3, 3, 9)

plt.bar(cwd['Country/Region'], cwd['Recovered']/cwd['Confirmed'])

plt.ylabel('Death percentage till 09/03/2020')

plt.xticks(rotation = 90)



plt.tight_layout()
italy_data_region = pd.read_csv("/kaggle/input/italy-data/covid19_italy_region.csv")

italy_daya_province = pd.read_csv("/kaggle/input/italy-data/covid19_italy_province.csv")
italy_data_region.head()
italy_daya_province.head()
import folium

italy_map = folium.Map(location=[42.8719,12.5674 ], zoom_start=5,tiles='Stamen Toner')



for lat, lon,RegionName,TotalPositiveCases,Recovered,Deaths,TotalHospitalizedPatients in zip(italy_data_region['Latitude'], italy_data_region['Longitude'],italy_data_region['RegionName'],italy_data_region['TotalPositiveCases'],italy_data_region['Recovered'],italy_data_region['Deaths'],italy_data_region['TotalHospitalizedPatients']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                      popup =('RegionName: ' + str(RegionName) + '<br>'

                    'TotalPositiveCases: ' + str(TotalPositiveCases) + '<br>'

                    'TotalHospitalizedPatients: ' + str(TotalHospitalizedPatients) + '<br>'

                      'Recovered: ' + str(Recovered) + '<br>'

                      'Deaths: ' + str(Deaths) + '<br>'),



                        fill_color='red',

                        fill_opacity=0.7 ).add_to(italy_map)

italy_map
italy_data_region.columns
italy_data_region.drop(['SNo', 'Country'], axis=1 , inplace=True)
italy_data_region['Date'] = pd.Series(pd.to_datetime(italy_data_region['Date'])).dt.date
date_wise_total = italy_data_region[['Date', 'TotalPositiveCases', 'NewPositiveCases', 'Recovered', 'Deaths']].groupby('Date').sum().reset_index()



plt.figure(figsize = (12, 6))

plt.subplot(121)

sns.barplot(date_wise_total['Date'], date_wise_total['TotalPositiveCases'])

plt.xticks(rotation = 'vertical')

plt.title("Total Positive Cases Day wise")



plt.subplot(122)

sns.barplot(date_wise_total['Date'], date_wise_total['NewPositiveCases'])

plt.xticks(rotation = 'vertical')

plt.title("Total Positive Cases Day wise")
from sklearn.linear_model import LinearRegression



x = np.array(list(range(1, len(date_wise_total)+1))).reshape(-1, 1)

y = date_wise_total['TotalPositiveCases']
model = LinearRegression()

model.fit(x, np.log(y))
prediction = model.predict(x)
plt.figure(figsize = (12, 6))

plt.subplot(121)

plt.scatter(x, np.log(y))

plt.plot(x,prediction)

plt.title("log transformation plot")

plt.xlabel("TotalPositiveCases")

plt.xlabel("Day")

plt.legend()



plt.subplot(122)

plt.scatter(x, y)

plt.plot(x,np.exp(prediction))

plt.title("Estimation plot")

plt.xlabel("TotalPositiveCases")

plt.xlabel("Day")

plt.legend()
next_ = np.array(list(range(21,31))).reshape(-1, 1)

prediction = np.exp(model.predict(next_))
x1 = np.array(list(range(1, len(date_wise_total['NewPositiveCases']) +1))).reshape(-1, 1)

y1 = date_wise_total['NewPositiveCases']





model2 = LinearRegression()

model2.fit(x1, np.log(y1))



prediction1 = np.exp(model2.predict(x1))



plt.scatter(x1, y1)

plt.plot(prediction1)

plt.title("New Positive Cases")
prediction2 = np.exp(model2.predict(next_))
print("If the situation continues the Total Positive Cases in Italy alone will be ", round(prediction[-1]))

print("New Positive Cases will be increased as", round(prediction2[-1]))