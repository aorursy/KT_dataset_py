import numpy as np

import pandas as pd

import requests

import folium

import os

import json

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.io.json import json_normalize



sns.set()
# read the csv file and name it as 'monthly'

# header=0 --> so that the first row (with all the categorical names, not values) is defined as the header of the table.

monthly = pd.read_csv('../input/seouls-air-quality-and-sources-of-pollution/seoul_monthly_average_air_pollution.csv', header=0)



# head() function shows the first five rows of the table. 

monthly.head()
daily = pd.read_csv('../input/seouls-air-quality-and-sources-of-pollution/seoul_daily_average_air_pollution.csv', header=0)

daily.head()
yearly = pd.read_csv('../input/seouls-air-quality-and-sources-of-pollution/seoul_yearly_average_air_pollution.csv', header=0)

yearly.head()
# Changing the column names of the 'yearly' data table

yearly.columns = ['Year', 'Place_Measured', 'Nitrogen_Dioxide(ppm)', 'Ozone(ppm)', 'Carbon_Monoxide(ppm)', 'Sulfur_Dioxide(ppm)', 'PM-10', 'PM-2.5']

yearly.head()
gu_list_eng = ['Gangnam-gu', 'Gangdong-gu', 'Gangbuk-gu', 'Gangseo-gu', 'Gwanak-gu', 'Gwangjin-gu', 'Guro-gu',

 'Geumcheon-gu', 'Nowon-gu', 'Dobong-gu', 'Dongdaemun-gu', 'Dongjak-gu', 'Mapo-gu',

 'Seodaemun-gu', 'Seocho-gu', 'Seongdong-gu', 'Seongbuk-gu', 'Songpa-gu', 'Yangcheon-gu', 'Yeongdeungpo-gu',

'Yongsan-gu', 'Eunpyeong-gu', 'Jongno-gu', 'Jung-gu', 'Jungnang-gu']



gu_list_kor = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구',

 '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구',

'용산구', '은평구', '종로구', '중구', '중랑구']



# The first item of gu_list_kor corresponds to the first item of gu_list_eng, and vice versa. 
for index, row in yearly.iterrows(): # iterrows() allows the for loop to go through each and every row.

    if yearly.iloc[index, 1] in gu_list_kor: # index is row number, the number 1 points to the column name 'Place_Measured'

        i = gu_list_kor.index(yearly.iloc[index, 1]) # Get the list index of the matching word

        yearly.iloc[index, 1] = gu_list_eng[i] #Find the corresponding word in the English list, and apply it.



yearly.head(20)
yearly = yearly.loc[yearly['Place_Measured'].isin(gu_list_eng)]



yearly.head()
yearly.to_csv('Processed_yearly_seoul_air_quality.csv')
print(yearly.Year.unique())
print(yearly.Place_Measured.unique())



# Number of districts

len(yearly.Place_Measured.unique())
#The maximum pm-10 level recored in history

print(max(yearly.iloc[:, 6]))



#The maximum pm-2.5 level recored in history

print(max(yearly.iloc[:, 7]))
nowon = yearly.loc[yearly.Place_Measured == 'Nowon-gu']



#Let's see the entire history

nowon
def lineplot_gu(figsize=(20, 12), x='Year', y='PM-10', gu=''):

    data = yearly.loc[yearly.Place_Measured == gu]

    plt.figure(figsize=figsize)

    sns.lineplot(x='Year', y='PM-10', data=data, markers=True)

    plt.xlabel(x)

    if y == 'PM-10' or y == 'PM-2.5':

        plt.ylabel(y + ' (μg/m3)')

    plt.xticks(range(min(data['Year']), max(data['Year']) + 1))

    plt.title('{} level in {} from {} to {}'.format(y, gu, min(data['Year']), max(data['Year'])))
#Let's visualize the level of PM-10 of Nowon-gu

lineplot_gu(gu='Nowon-gu')
lineplot_gu(y='Ozone', gu='Gangnam-gu')
monthly.head()
daily.head()
monthly.columns = ['Time_Measured', 'Place_Measured', 'Nitrogen_Dioxide(ppm)', 'Ozone(ppm)', 'Carbon_Monoxide(ppm)', 'Sulfur_Dioxide(ppm)', 'PM-10', 'PM-2.5']

daily.columns = ['Time_Measured', 'Place_Measured', 'Nitrogen_Dioxide(ppm)', 'Ozone(ppm)', 'Carbon_Monoxide(ppm)', 'Sulfur_Dioxide(ppm)', 'PM-10', 'PM-2.5']
# For Monthly table.



for index, row in monthly.iterrows():

    if monthly.iloc[index, 1] in gu_list_kor:

        i = gu_list_kor.index(monthly.iloc[index, 1])

        monthly.iloc[index, 1] = gu_list_eng[i]



monthly = monthly.loc[monthly['Place_Measured'].isin(gu_list_eng)]

monthly.to_csv('Processed_monthly_seoul_air_quality.csv')

monthly.head()
# For Daily table.



for index, row in daily.iterrows():

    if daily.iloc[index, 1] in gu_list_kor:

        i = gu_list_kor.index(daily.iloc[index, 1])

        daily.iloc[index, 1] = gu_list_eng[i]



daily = daily.loc[daily['Place_Measured'].isin(gu_list_eng)]

daily.to_csv('Processed_daily_seoul_air_quality.csv')

daily.head()
factory_emissions = pd.read_csv('../input/seouls-air-quality-and-sources-of-pollution/report_on_factory-emissions.txt', sep=' ', header=None)

factory_emissions.head(10)
# Use \t as the separator

factories_emissions = pd.read_csv('../input/seouls-air-quality-and-sources-of-pollution/report_on_factory-emissions.txt', sep='\t', header=None)

factories_emissions.head(10)
factories_emissions = factories_emissions.loc[2:, :] #Get rid of the labels on the top two rows.

factories_emissions.head()
factories_emissions.columns = ['Year', 'District', 'Total', 'Category1', 'Category2', 'Category3', 'Category4', 'Category5']

factories_emissions.drop(factories_emissions.index[0], inplace=True) #Remove the first row, as it was supposed to be the labels.

factories_emissions.head()
#Now let's remove the rows that show the total of factories in every year.

factories_emissions = factories_emissions.loc[factories_emissions['District'] != '합계']

factories_emissions.head()
factories_emissions.reset_index(drop=True, inplace=True)

factories_emissions.head()
for index, row in factories_emissions.iterrows():

    if factories_emissions.iloc[index, 1] in gu_list_kor:

        i = gu_list_kor.index(factories_emissions.iloc[index, 1])

        factories_emissions.iloc[index, 1] = gu_list_eng[i]



#Save the data table as a separate file.

factories_emissions.to_csv('Processed_factories_emissions_seoul.csv')

factories_emissions.head()
print(factories_emissions.Year.unique())
companies_info = pd.read_csv('../input/seoul-air-qualit-data-extended/seoul_pollutants_companies_information.csv', header=0)

companies_info.head()
companies_info.columns
companies_info.drop(['번호', '영업상태명', '폐업일자', '휴업시작일자', '휴업종료일자', '재개업일자', '소재지면적', '방지시설연간가동일수', '방지시설조업시간', '배출시설연간가동일수', '배출시설조업시간', '주생산품명', '소재지우편번호', '상세영업상태명', '전화번호', '환경업무구분명', '인허가번호'], 1, inplace=True)

#inplace=true is used to change the data table directly. If it is omitted, the original data table remains as it is. 
# The updated table.

companies_info.head()
companies_info.columns = ['Name', 'Location_Address', 'Street_Address', 'Authorization_Date', 'Industry', 'Category', 'Location_X', 'Location_Y']

companies_info.head()
# Save the table as a separate file.

companies_info.to_csv('Processed_seoul_pollutants_companies_information.csv')
pollution_monitors = pd.read_csv('../input/seoul-air-qualit-data-extended/seoul_air_pollution_monitors_information.csv', header=0)

pollution_monitors.head()
pollution_monitors.columns = ['Monitor_Code', 'Monitor_Name', 'Address', 'Order', 'Authorized_Code']

pollution_monitors = pollution_monitors.iloc[:25, :]

pollution_monitors.shape
for index, row in pollution_monitors.iterrows():

    if pollution_monitors.iloc[index, 1] in gu_list_kor:

        i = gu_list_kor.index(pollution_monitors.iloc[index, 1])

        pollution_monitors.iloc[index, 1] = gu_list_eng[i]



pollution_monitors.to_csv('Processed_seoul_air_pollution_monitors_information.csv')

pollution_monitors.head()