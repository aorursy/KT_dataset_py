#import essential libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import os

np.random.seed(0)

print("Libraries Imported, Setup Complete")
# import dataset



data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")



# view the top and bottom of the dataset



data.head(5)

data.tail(5)



# view a sample of the dataset 



data.sample(10)
# view the column names in the dataset



data.columns
# change column names as per your suiting



data = data.rename(columns={'country':'Country', 'year':'Year', 'sex':'Gender', 'age':'Age', 'suicides_no':'SuicideNumber', 'population':'Population', 'suicides/100k pop':'SuicidePer100K', 'country-year':'CountryYearCode', 'HDI for year':'HDI',' gdp_for_year ($) ':'GDP', 'gdp_per_capita ($)':'GDPCapita', 'generation':'Generation'})

data.head(5)
# check for any empty values



data.isnull().any()
# find the number of empty values in the dataset



data.isnull().sum()
# drop column in dataset



data = data.drop(['HDI', 'CountryYearCode'], axis=1)
# converting all string numerals to int values



data['GDP'].replace(',','', regex=True, inplace=True)

data["GDP"] = data["GDP"].astype('int64')

data["SuicideNumber"] = data["SuicideNumber"].astype(int)

data["Population"] = data["Population"].astype(int)

data["GDPCapita"] = data["GDPCapita"].astype(int)

data["SuicidePer100K"] = data["SuicidePer100K"].astype(float)

data["SuicideNumber"] = data["SuicideNumber"].astype(int)



data
# checking for unique values within the Year column



year = data['Year'].unique()

year
# gathering data from SuicideNumber and Population based on individual years 



SuicideVal = []

Population = []

suicide_percentage = []

for i in year:

    SuicideVal.append(sum(data[data['Year'] == i]['SuicideNumber']))

    Population.append(sum(data[data['Year'] == i]['Population']))

billion = 1000000000

Population[:] = [x / billion for x in Population]

Population[:] = [round(x, 3) for x in Population]

print(SuicideVal)

print(Population)
# forming a dataset from the gathered data



year = data.Year.unique()

pop_suicide_data = {'Year': year, 'Suicides': SuicideVal, 'Population in Billions': Population}

pop_suicide = pd.DataFrame(data=pop_suicide_data)

pop_suicide = pop_suicide.drop(index=31)

pop_suicide = pop_suicide.set_index('Year')

pop_suicide
# plotting a line graph depicting the worldwide population per year



fig, ax = plt.subplots(figsize=(20,7))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_popyr = sns.lineplot(x='Year', y='Population in Billions', data=pop_suicide.reset_index(), label="Worldwide Population", markers=True, ax=ax)

ax.set_title("Worldwide Population Per Year", fontsize=20)

ax.set_ylabel("Population (in Billions)")

ax.set_xlabel("Year")

ax.set_xticks(year)

ax.set_xticklabels(year, rotation=45)

sns_popyr.figure.savefig("Worldwide Population Per Year.png")

plt.show()
# plotting a line graph depicting the number of suicides per year



fig, ax = plt.subplots(figsize=(20,7))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_suiyr = sns.lineplot(x='Year', y='Suicides', data=pop_suicide.reset_index(), label="Number of Suicides", markers=True, ax=ax)

ax.set_title("Number of Suicides Per Year", fontsize=20)

ax.set_ylabel("Number of Suicides")

ax.set_xlabel("Year")

ax.set_xticks(year)

ax.set_xticklabels(year, rotation=45)

sns_suiyr.figure.savefig("Number of Suicides Per Year.png")

plt.show()
# plotting a bar plot comparing the number of suicides to the worldwide population



fig, ax = plt.subplots(figsize=(20,7))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns.set_style("white")

sns_suipop = sns.barplot(x='Population in Billions', y='Suicides', data=pop_suicide.reset_index(), label="Number of Suicides vs Worldwide Population", ax=ax)

ax.set_title("Number of Suicides vs Worldwide Population", fontsize=20)

ax.set_ylabel("Number of Suicides")

ax.set_xlabel("Population (in Billions)")

ax.set_xticklabels(Population, rotation=45)

sns_suipop.figure.savefig("Number of Suicides vs Worldwide Population.png")

plt.show()
# checking for unique values within the Country column



country = data.Country.unique()

len(country)
# gather all the necessary data from the dataset based on individual countries



GDPperCap = []

SuicideNo = []

Suicidesper100K = []

SuicideAvg = []

for i in country:

    GDPperCap.append(sum(data[data.Country == i]['GDPCapita']))

    Suicidesper100K.append(sum(data[data.Country == i]['SuicidePer100K']))

    SuicideNo.append(sum(data[data.Country == i]['SuicideNumber']))

    SuicideAvg.append(sum(data[data.Country == i]['SuicideNumber']))

GDPperCap[:] = [x/180 for x in GDPperCap]

GDPperCap[:] = [round(x, 3) for x in GDPperCap]

Suicidesper100K[:] = [x/12 for x in Suicidesper100K]

Suicidesper100K[:] = [round(x, 3) for x in Suicidesper100K]

SuicideNo[:] = [x for x in SuicideNo]

SuicideNo[:] = [int(x) for x in SuicideNo]

SuicideAvg[:] = [x/30 for x in SuicideAvg]

SuicideAvg[:] = [round(x, 2) for x in SuicideAvg]

print(GDPperCap)

print(Suicidesper100K)

print(SuicideNo)

print(SuicideAvg)
# create a dataset based on the data collected



country_suicide_data = {'Country': country, 'GDPperCapita': GDPperCap, 'SuicideNumber': SuicideNo, 'SuicideAverage': SuicideAvg, 'Suicideper100K': Suicidesper100K}

country_suicide = pd.DataFrame(data=country_suicide_data)

country_suicide = country_suicide.drop(index=27)

country_suicide = country_suicide.drop(index=59)

country_suicide = country_suicide.drop(index=76)

country_suicide = country_suicide.sort_values(by=['SuicideNumber'], ascending=False)

country_suicide
# creating a bar graph comparing the GDP per Capita of each country



fig, ax = plt.subplots(figsize=(15,30))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_gdpcapcon = sns.barplot(x='GDPperCapita', y='Country', data=country_suicide.reset_index(), label="GDP Per Capita Per Country", ax=ax)

ax.set_title("GDP Per Capita Per Country", fontsize=20)

ax.set_ylabel("Countries (in Descending order of Number of Suicides)")

ax.set_xlabel("GDP Per Capita")

sns_gdpcapcon.figure.savefig("GDP Per Capita Per Country.png")

plt.show()
# creating a bar graph to plot the number of suicides per 100K population per country



fig, ax = plt.subplots(figsize=(15,30))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_suipopcon = sns.barplot(x='Suicideper100K', y='Country', data=country_suicide.reset_index(), label="Number of Suicides Per 100K Population Per Country", ax=ax)

ax.set_title("Number of Suicides Per 100K Population Per Country", fontsize=20)

ax.set_ylabel("Countries (in Descending order of Number of Suicides)")

ax.set_xlabel("Number of Suicides per 100K Population")

sns_suipopcon.figure.savefig("Number of Suicides Per 100K Population Per Country.png")

plt.show()
# calculating the total number of male and female suicides



suicide_male = 0

suicide_female = 0

for i in year:

    suicide_male = suicide_male + sum(data[(data.Year == i) & (data.Gender == 'male')]['SuicideNumber'])

    suicide_female = suicide_female + sum(data[(data.Year == i) & (data.Gender == 'female')]['SuicideNumber'])

print(suicide_male)

print(suicide_female)
# plotting a pie chart depicting the ratio of male to female suicides



labels = ['Male Suicides','Female Suicides']

values = [suicide_male, suicide_female]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1])])

fig.update_layout(title_text="Ratio of Male Suicides to Female Suicides")

fig.show()
# calculating the total number of suicides by generation



generation = data.Generation.unique()

gen = []

male_gen = []

female_gen = []

for i in generation:

    gen.append(sum(data[(data.Generation == i)]['SuicideNumber']))

    male_gen.append(sum(data[(data.Generation == i) & (data.Gender == 'male')]['SuicideNumber']))

    female_gen.append(sum(data[(data.Generation == i) & (data.Gender == 'female')]['SuicideNumber']))

print(gen)

print(male_gen)

print(female_gen)
# plotting a pie chart depicting the ratio of suicides among different generations



labels = generation

values = gen



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])])

fig.update_layout(title_text="Division of Suicides by Generation")

fig.show()
# plot a pie chart depicting the number of male suicides across various generations



labels = generation

values = male_gen



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])])

fig.update_layout(title_text="Division of Suicides by Generation - Male")

fig.show()
# plot a pie chart depicting the number of male suicides across various generations



labels = generation

values = female_gen



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])])

fig.update_layout(title_text="Division of Suicides by Generation - Female")

fig.show()
# calculating total number of suicides by age demographic



age = data.Age.unique()

agedem = []

male_age = []

female_age = []

for i in age:

    agedem.append(sum(data[(data.Age == i)]['SuicideNumber']))

    male_age.append(sum(data[(data.Age == i) & (data.Gender == 'male')]['SuicideNumber']))

    female_age.append(sum(data[(data.Age == i) & (data.Gender == 'female')]['SuicideNumber']))

print(agedem)

print(male_age)

print(female_age)
# plotting a pie chart depicting the ratio of suicides among different age demographics



labels = age

values = agedem



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])])

fig.update_layout(title_text="Division of Suicides by Age Demographic")

fig.show()
# plotting a pie chart depicting the ratio of male suicides among different age demographics



labels = age

values = male_age



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])])

fig.update_layout(title_text="Division of Suicides by Age Demographic - Male")

fig.show()
# plotting a pie chart depicting the ratio of female suicides among different age demographics



labels = age

values = female_age



fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2])])

fig.update_layout(title_text="Division of Suicides by Age Demographic - Female")

fig.show()
# segregating data based on year



yrMsuicide = []

yrFsuicide = []

yr_genx = []

yr_sil = []

yr_gigen = []

yr_boom = []

yr_mil = []

yr_genz = []

yr_5to14 = []

yr_15to24 = []

yr_25to34 = []

yr_35to54 = []

yr_55to74 = []

yr_75plus = []

for i in year:

    yrMsuicide.append(sum(data[(data.Year == i) & (data.Gender == 'male')]['SuicideNumber']))

    yrFsuicide.append(sum(data[(data.Year == i) & (data.Gender == 'female')]['SuicideNumber']))

    yr_genx.append(sum(data[(data.Year == i) & (data.Generation == 'Generation X')]['SuicideNumber']))

    yr_sil.append(sum(data[(data.Year == i) & (data.Generation == 'Silent')]['SuicideNumber']))

    yr_gigen.append(sum(data[(data.Year == i) & (data.Generation == 'G.I. Generation')]['SuicideNumber']))

    yr_boom.append(sum(data[(data.Year == i) & (data.Generation == 'Boomers')]['SuicideNumber']))

    yr_mil.append(sum(data[(data.Year == i) & (data.Generation == 'Millenials')]['SuicideNumber']))

    yr_genz.append(sum(data[(data.Year == i) & (data.Generation == 'Generation Z')]['SuicideNumber']))

    yr_5to14.append(sum(data[(data.Year == i) & (data.Age == '5-14 years')]['SuicideNumber']))

    yr_15to24.append(sum(data[(data.Year == i) & (data.Age == '15-24 years')]['SuicideNumber']))

    yr_25to34.append(sum(data[(data.Year == i) & (data.Age == '25-34 years')]['SuicideNumber']))

    yr_35to54.append(sum(data[(data.Year == i) & (data.Age == '35-54 years')]['SuicideNumber']))

    yr_55to74.append(sum(data[(data.Year == i) & (data.Age == '55-74 years')]['SuicideNumber']))

    yr_75plus.append(sum(data[(data.Year == i) & (data.Age == '75+ years')]['SuicideNumber']))



print(yrMsuicide)

print(yrFsuicide)

print(yr_genx)

print(yr_sil)

print(yr_gigen)

print(yr_boom)

print(yr_mil)

print(yr_genz)

print(yr_5to14)

print(yr_15to24)

print(yr_25to34)

print(yr_35to54)

print(yr_55to74)

print(yr_75plus)
# creating dataset to be used for analysis



genyr_suicide_data = {'Year':year, 'Generation X':yr_genx, 'Silent':yr_sil, 'G.I. Generation':yr_gigen, 'Boomers':yr_boom, 'Millenials':yr_mil, 'Generation Z':yr_genz}

genyr_suicide = pd.DataFrame(data=genyr_suicide_data)

sexyr_suicide_data = {'Year':year, 'Male Suicides':yrMsuicide, 'Female Suicides':yrFsuicide}

sexyr_suicide = pd.DataFrame(data=sexyr_suicide_data)

ageyr_suicide_data = {'Year':year, '5-14':yr_5to14, '15-24':yr_15to24, '25-34':yr_25to34, '35-54':yr_35to54, '55-74':yr_55to74, '75+':yr_75plus}

ageyr_suicide = pd.DataFrame(data=ageyr_suicide_data)

genyr_suicide = pd.melt(genyr_suicide, id_vars="Year", var_name="Generation", value_name="Suicides")

sexyr_suicide = pd.melt(sexyr_suicide, id_vars="Year", var_name="Gender", value_name="Suicides")

ageyr_suicide = pd.melt(ageyr_suicide, id_vars="Year", var_name="Age", value_name="Suicides")

ageyr_suicide = ageyr_suicide[ageyr_suicide.Year != 2016]

genyr_suicide = genyr_suicide[genyr_suicide.Year != 2016]

sexyr_suicide = sexyr_suicide[sexyr_suicide.Year != 2016]
# plotting a line graph comparing number of suicides per year across two genders



fig, ax = plt.subplots(figsize=(20,7))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_suisex = sns.lineplot(x='Year', y='Suicides', hue='Gender', data=sexyr_suicide.reset_index(), markers=True, ax=ax)

ax.set_title("Number of Suicides Per Year, Worldwide (Gender)", fontsize=20)

ax.set_ylabel("Number of Suicides")

ax.set_xlabel("Year")

ax.set_xticks(year)

ax.set_xticklabels(year, rotation=45)

sns_suisex.figure.savefig("Number of Suicides Per Year, Worldwide (Gender).png")

plt.show()
# plotting a line graph comparing number of suicides per year across all age demographics



fig, ax = plt.subplots(figsize=(20,7))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_suiage = sns.lineplot(x='Year', y='Suicides', hue='Age', data=ageyr_suicide.reset_index(), markers=True, ax=ax)

ax.set_title("Number of Suicides Per Year, Worldwide (Age Demographic)", fontsize=20)

ax.set_ylabel("Number of Suicides")

ax.set_xlabel("Year")

ax.set_xticks(year)

ax.set_xticklabels(year, rotation=45)

sns_suiage.figure.savefig("Number of Suicides Per Year, Worldwide (Age Demographic).png")

plt.show()
# plotting a line graph comparing number of suicides per year across all generations



fig, ax = plt.subplots(figsize=(20,7))

sns.despine()

sns.set_style("white")



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['text.color'] = '#000000'

plt.rcParams['axes.labelcolor']= '#000000'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['ytick.color'] = '#000000'

plt.rcParams['font.size']=15



sns_suigen = sns.lineplot(x='Year', y='Suicides', hue='Generation', data=genyr_suicide.reset_index(), markers=True, ax=ax)

ax.set_title("Number of Suicides Per Year, Worldwide (Generation)", fontsize=20)

ax.set_ylabel("Number of Suicides")

ax.set_xlabel("Year")

ax.set_xticks(year)

ax.set_xticklabels(year, rotation=45)

sns_suigen.figure.savefig("Number of Suicides Per Year, Worldwide (Generation).png")

plt.show()
# segregating data based on country



ctrMsuicide = []

ctrFsuicide = []

ctr_genx = []

ctr_sil = []

ctr_gigen = []

ctr_boom = []

ctr_mil = []

ctr_genz = []

ctr_5to14 = []

ctr_15to24 = []

ctr_25to34 = []

ctr_35to54 = []

ctr_55to74 = []

ctr_75plus = []

for i in country:

    ctrMsuicide.append(sum(data[(data.Country == i) & (data.Gender == 'male')]['SuicideNumber']))

    ctrFsuicide.append(sum(data[(data.Country == i) & (data.Gender == 'female')]['SuicideNumber']))

    ctr_genx.append(sum(data[(data.Country == i) & (data.Generation == 'Generation X')]['SuicideNumber']))

    ctr_sil.append(sum(data[(data.Country == i) & (data.Generation == 'Silent')]['SuicideNumber']))

    ctr_gigen.append(sum(data[(data.Country == i) & (data.Generation == 'G.I. Generation')]['SuicideNumber']))

    ctr_boom.append(sum(data[(data.Country == i) & (data.Generation == 'Boomers')]['SuicideNumber']))

    ctr_mil.append(sum(data[(data.Country == i) & (data.Generation == 'Millenials')]['SuicideNumber']))

    ctr_genz.append(sum(data[(data.Country == i) & (data.Generation == 'Generation Z')]['SuicideNumber']))

    ctr_5to14.append(sum(data[(data.Country == i) & (data.Age == '5-14 years')]['SuicideNumber']))

    ctr_15to24.append(sum(data[(data.Country == i) & (data.Age == '15-24 years')]['SuicideNumber']))

    ctr_25to34.append(sum(data[(data.Country == i) & (data.Age == '25-34 years')]['SuicideNumber']))

    ctr_35to54.append(sum(data[(data.Country == i) & (data.Age == '35-54 years')]['SuicideNumber']))

    ctr_55to74.append(sum(data[(data.Country == i) & (data.Age == '55-74 years')]['SuicideNumber']))

    ctr_75plus.append(sum(data[(data.Country == i) & (data.Age == '75+ years')]['SuicideNumber']))



print(ctrMsuicide)

print(ctrFsuicide)

print(ctr_genx)

print(ctr_sil)

print(ctr_gigen)

print(ctr_boom)

print(ctr_mil)

print(ctr_genz)

print(ctr_5to14)

print(ctr_15to24)

print(ctr_25to34)

print(ctr_35to54)

print(ctr_55to74)

print(ctr_75plus)
# creating dataset to be used for analysis



gencon_suicide_data = {'Country':country, 'Generation X':ctr_genx, 'Silent':ctr_sil, 'G.I. Generation':ctr_gigen, 'Boomers':ctr_boom, 'Millenials':ctr_mil, 'Generation Z':ctr_genz}

gencon_suicide = pd.DataFrame(data=gencon_suicide_data)

sexcon_suicide_data = {'Country':country, 'Male Suicides':ctrMsuicide, 'Female Suicides':ctrFsuicide}

sexcon_suicide = pd.DataFrame(data=sexcon_suicide_data)

agecon_suicide_data = {'Country':country, '5-14':ctr_5to14, '15-24':ctr_15to24, '25-34':ctr_25to34, '35-54':ctr_35to54, '55-74':ctr_55to74, '75+':ctr_75plus}

agecon_suicide = pd.DataFrame(data=agecon_suicide_data)

gencon_suicide = pd.melt(gencon_suicide, id_vars="Country", var_name="Generation", value_name="Suicides")

sexcon_suicide = pd.melt(sexcon_suicide, id_vars="Country", var_name="Gender", value_name="Suicides")

agecon_suicide = pd.melt(agecon_suicide, id_vars="Country", var_name="Age", value_name="Suicides")

gencon_suicide = gencon_suicide.sort_values('Suicides', ascending = True)

sexcon_suicide = sexcon_suicide.sort_values('Suicides', ascending = True)

agecon_suicide = agecon_suicide.sort_values('Suicides', ascending = True)
# plotting a bar graph depicting number of suicides per country across genders



fig = px.bar(sexcon_suicide, x="Suicides", y="Country", color="Gender", barmode="relative")



fig.update_layout(legend_title_text = "Gender", width=1000, height=2000, plot_bgcolor='#fff', title="Number of Suicides Per Country (Gender)")

fig.update_xaxes(title_text="Countries")

fig.update_yaxes(title_text="Number of Suicides")

fig.show()
# plotting a bar graph depicting number of suicides per country across age demographic



fig = px.bar(agecon_suicide, x="Suicides", y="Country", color="Age", barmode="relative")



fig.update_layout(legend_title_text = "Age Demographic", width=1000, height=2500, plot_bgcolor='#fff', title="Number of Suicides Per Country (Age Demographic)")

fig.update_xaxes(title_text="Countries")

fig.update_yaxes(title_text="Number of Suicides")

fig.show()
# plotting a bar graph depicting number of suicides per country across generation



fig = px.bar(gencon_suicide, x="Suicides", y="Country", color="Generation", barmode="relative")



fig.update_layout(legend_title_text = "Generation", width=1000, height=2500, plot_bgcolor='#fff', title="Number of Suicides Per Country (Generation)")

fig.update_xaxes(title_text="Countries")

fig.update_yaxes(title_text="Number of Suicides")

fig.show()