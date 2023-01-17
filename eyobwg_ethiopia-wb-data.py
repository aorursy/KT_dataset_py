# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# Any results you write to the current directory are saved as output.
et_data = pd.read_csv('../input/world_bank_ETH_data.csv')



#We are just interested in the data after 2000

selected_fields = ['Indicator Name', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

et_data = et_data[selected_fields]



#et_data = et_data.iloc[10:20, :]



#Set 'Indicator Name' as an index

et_data = et_data.set_index("Indicator Name")

gdp_per_capita = et_data.loc["GDP per capita (constant LCU)", :]

male_industry_employment = et_data.loc["Employment in industry, male (% of male employment) (modeled ILO estimate)", :]

female_industry_employment = et_data.loc["Employment in industry, female (% of female employment) (modeled ILO estimate)", :]

plt.figure(figsize=(12,6))

plt.plot(gdp_per_capita.keys(), gdp_per_capita)

plt.xlabel('Year')

plt.title("GDP per capita (constant 2011 PPP $)")

plt.grid(True)

plt.show()
#Unemployment, total (% of total labor force) (modeled ILO estimate)

unemployment = et_data.loc['Unemployment, total (% of total labor force) (modeled ILO estimate)']

unemployment_male = et_data.loc['Unemployment, male (% of male labor force) (modeled ILO estimate)']

unemployment_female = et_data.loc['Unemployment, female (% of female labor force) (modeled ILO estimate)']



plt.figure(figsize=(12,6))



plt.plot(unemployment.keys(), unemployment)

plt.plot(unemployment.keys(), unemployment_male, '-', label='male')

plt.plot(unemployment.keys(), unemployment_female, '--', label='female')

plt.xlabel('Year')

plt.title("Unemployment, total (% of total labor force) (modeled ILO estimate)")

plt.grid(True)

plt.show()
access_to_electricity = et_data.loc['Access to electricity (% of population)']



plt.figure(figsize=(12,6))

plt.plot(access_to_electricity.keys(), access_to_electricity)

plt.xlabel('Year')

plt.title("Access to electricity (% of population))")

plt.grid(True)

plt.show()
#Access to electricity, rural and urban

access_to_electricity_urban = et_data.loc['Access to electricity, urban (% of urban population)']

access_to_electricity_rural = et_data.loc['Access to electricity, rural (% of rural population)']

access_to_electricity = et_data.loc['Access to electricity (% of population)']



plt.figure(figsize=(12,6))



plt.plot(access_to_electricity.keys(), access_to_electricity)

plt.plot(access_to_electricity.keys(), access_to_electricity_urban, '-', label='access_to_electricity_urban')

plt.plot(access_to_electricity.keys(), access_to_electricity_rural, '--', label='access_to_electricity_rural')

plt.xlabel('Year')

plt.title("Access to electricity, rural and urban")

plt.grid(True)

plt.show()
trade_data = et_data.loc['Trade (% of GDP)']

trade_data.describe()
good_imports = et_data.loc['Goods imports (BoP, current US$)']



plt.figure(figsize=(12,6))

plt.plot(good_imports.keys(), good_imports)

plt.xlabel('Year')

plt.title("Goods imports (BoP, current US$)")

plt.grid(True)

plt.show()
gender_equality = et_data.loc['CPIA gender equality rating (1=low to 6=high)']

gender_equality
ICT_goods_imports = et_data.loc['ICT goods imports (% total goods imports)']



plt.figure(figsize=(12,6))

plt.plot(ICT_goods_imports.keys(), ICT_goods_imports)

plt.xlabel('Year')

plt.title("ICT goods imports (% total goods imports)")

plt.grid(True)

plt.show()
physicians = et_data.loc['Physicians (per 1,000 people)']

nurses = et_data.loc['Nurses and midwives (per 1,000 people)']



plt.figure(figsize=(12,6))

plt.plot(physicians.keys(), physicians)

plt.plot(nurses.keys(), nurses,'--')

plt.xlabel('Year')

plt.title("Physicians and Nurses (per 1,000 people)")

plt.grid(True)

plt.show()
#Number of maternal deaths

maternal_deaths = et_data.loc['Number of maternal deaths']

physicians_data = et_data.loc['Physicians (per 1,000 people)']



plt.figure(figsize=(12,6))

plt.plot(maternal_deaths.keys(), maternal_deaths, '-')

#plt.plot(maternal_deaths.keys(), physicians_data, '--')

plt.xlabel('Year')

plt.title("Number of maternal deaths")

plt.grid(True)

plt.show()
#Individuals using the Internet (% of population)

individual_internet_users = et_data.loc['Individuals using the Internet (% of population)']



plt.figure(figsize=(12,6))

plt.plot(individual_internet_users.keys(), individual_internet_users)



plt.xlabel('Year')

plt.title("Individuals using the Internet (% of population)")

plt.grid(True)

plt.show()
#Mobile cellular subscriptions (per 100 people)

mobile_subscription = et_data.loc['Mobile cellular subscriptions (per 100 people)']



plt.figure(figsize=(12,6))

plt.plot(mobile_subscription.keys(), mobile_subscription)



plt.xlabel('Year')

plt.title("Mobile cellular subscriptions (per 100 people)")

plt.grid(True)

plt.show()
#Children out of school

children_out_of_school_male = et_data.loc['Children out of school, male (% of male primary school age)']

children_out_of_school_female = et_data.loc['Children out of school, female (% of female primary school age)']



plt.figure(figsize=(12,6))

plt.plot(children_out_of_school_male.keys(), children_out_of_school_male, '-')

plt.plot(children_out_of_school_female.keys(), children_out_of_school_female, '--')

plt.xlabel('Year')

plt.title("Children out of school (% of male and female primary school age)")

plt.grid(True)

plt.show()
over_age_students = et_data.loc['Over-age students, primary (% of enrollment)']



plt.figure(figsize=(12,6))

plt.plot(over_age_students.keys(), over_age_students)

plt.xlabel('Year')

plt.title("Over-age students, primary (% of enrollment)")

plt.grid(True)

plt.show()
#Time required to start a business,(days)

time_to_start_business_male = et_data.loc['Time required to start a business, male (days)']

time_to_start_business_female = et_data.loc['Time required to start a business, female (days)']



plt.figure(figsize=(12,6))

plt.plot(time_to_start_business_male.keys(), time_to_start_business_male, '-')

plt.plot(time_to_start_business_female.keys(), time_to_start_business_female, '--')

plt.xlabel('Year')

plt.title("Time required to start a business,(days)")

plt.grid(True)

plt.show()
#Agricultural irrigated land (% of total agricultural land)

agricultural_irrigated_land = et_data.loc['Agricultural irrigated land (% of total agricultural land)']

#children_out_of_school_female = et_data.loc['Children out of school, female (% of female primary school age)']



plt.figure(figsize=(12,6))

plt.plot(agricultural_irrigated_land.keys(), agricultural_irrigated_land, '-')

plt.xlabel('Year')

plt.title("Agricultural irrigated land (% of total agricultural land)")

plt.grid(True)

plt.show()
# Agricultural irrigated land (% of total agricultural land)

forest_area = et_data.loc['Forest area (% of land area)']

#children_out_of_school_female = et_data.loc['Children out of school, female (% of female primary school age)']



plt.figure(figsize=(12,6))

plt.plot(forest_area.keys(), forest_area)

plt.xlabel('Year')

plt.title("Forest area (% of land area)")

plt.grid(True)

plt.show()
# Agricultural irrigated land (% of total agricultural land)

agricultural_land = et_data.loc['Agricultural land (% of land area)']





plt.figure(figsize=(12,6))

plt.plot(agricultural_land.keys(), agricultural_land)

plt.xlabel('Year')

plt.title("Agricultural land (% of land area)")

plt.grid(True)

plt.show()
# Methodology assessment of statistical capacity (scale 0 - 100)

statistical_capacity = et_data.loc['Methodology assessment of statistical capacity (scale 0 - 100)']





plt.figure(figsize=(12,6))

plt.plot(statistical_capacity.keys(), statistical_capacity)

plt.xlabel('Year')

plt.title("Methodology assessment of statistical capacity (scale 0 - 100)")

plt.grid(True)

plt.show()
assessment_of_statistical_capacity = et_data.loc['Source data assessment of statistical capacity (scale 0 - 100)']



plt.figure(figsize=(12,6))

plt.plot(assessment_of_statistical_capacity.keys(), assessment_of_statistical_capacity)

plt.xlabel('Year')

plt.title("Source data assessment of statistical capacity (scale 0 - 100)")

plt.grid(True)

plt.show()
# Statistical Capacity score (Overall average)

statistical_capacity_score = et_data.loc['Statistical Capacity score (Overall average)']





plt.figure(figsize=(12,6))

plt.plot(statistical_capacity_score.keys(), statistical_capacity_score)

plt.xlabel('Year')

plt.title("Statistical Capacity score (Overall average)")

plt.grid(True)

plt.show()
air_transport = et_data.loc['Air transport, passengers carried']



plt.figure(figsize=(12,6))

plt.plot(air_transport.keys(), air_transport)

plt.xlabel('Year')

plt.title("Air transport, passengers carried")

plt.grid(True)

plt.show()
quality_of_port_infrastructure = et_data.loc['Quality of port infrastructure, WEF (1=extremely underdeveloped to 7=well developed and efficient by international standards)']



plt.figure(figsize=(12,6))

plt.plot(quality_of_port_infrastructure.keys(), quality_of_port_infrastructure)

plt.xlabel('Year')

plt.title("Quality of port infrastructure, WEF (1=extremely underdeveloped to 7=well developed and efficient by international standards)")

plt.grid(True)

plt.show()
scientific_and_technical_journal_articles = et_data.loc['Scientific and technical journal articles']



plt.figure(figsize=(12,6))

plt.plot(scientific_and_technical_journal_articles.keys(), scientific_and_technical_journal_articles)

plt.xlabel('Year')

plt.title("Scientific and technical journal articles")

plt.grid(True)

plt.show()