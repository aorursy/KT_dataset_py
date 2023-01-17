import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
data_suicide = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

data_suicide.head()
data_suicide['age_range'] = data_suicide['age'].map(lambda x:x.split(' ')[0])

data_suicide = data_suicide[data_suicide['year'] < 2016]
data_grp_mf = data_suicide.groupby(by=['sex']).agg({"suicides_no": ['sum']})

data_grp_mf.columns = ['total_suicide']

data_grp_mf.reset_index(inplace=True)

# data_grp_mf



fig1, ax1 = plt.subplots(figsize=(15, 7))

wedges, texts, autotexts = ax1.pie(data_grp_mf['total_suicide'], labels=data_grp_mf['sex'], autopct='%1.1f%%', startangle=90, colors=['#fd5235', '#636efa'])

ax1.axis('equal')



plt.setp(autotexts, size=15, color="w")

plt.figtext(.5,.91,'Worldwide Suicide by Gender (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_sc_no = data_suicide.groupby(by=['age_range', 'sex']).agg({'suicides_no': ['sum']})

data_sc_no.columns = ['total_suicide']

data_sc_no.reset_index(inplace=True)

# data_sc_no



color = sns.color_palette("Set2")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='age_range', y='total_suicide', hue='sex', data=data_sc_no, palette=color)



plt.setp(autotexts, size=10, color="w")

plt.figtext(.5,.91,'Worldwide Suicide on total suicide by Age (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_year_no = data_suicide.groupby(by=['year', 'sex']).agg({'suicides_no': ['sum']})

data_year_no.columns = ['total_suicide']

data_year_no.reset_index(inplace=True)

data_year_no



color = sns.color_palette("Set2")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='year', y='total_suicide', hue='sex', data=data_year_no, palette=color)

plt.xticks(rotation=90)



plt.figtext(.5,.91,'Worldwide Suicide by Year (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_popk_year = data_suicide.groupby(by=['year']).agg({'suicides_no':['sum'], 'population': ['sum']})

data_popk_year.columns = ['total_suicide', 'total_population']

data_popk_year = data_popk_year.reset_index()



data_popk_year['deth_rat_100k'] = data_popk_year['total_suicide'] / data_popk_year['total_population'] * 100000

# data_popk_year

plt.figure(figsize=(15, 10))

sns.lineplot(x="year", y="deth_rat_100k", marker='o', markersize=10, color='#f77189', data=data_popk_year)



plt.figtext(.5,.91,'Worldwide Suicide on 100k pop rate by Year (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_popk_year_sex = data_suicide.groupby(by=['year', 'sex']).agg({'suicides_no':['sum'], 'population': ['sum']})

data_popk_year_sex.columns = ['total_suicide', 'total_population']

data_popk_year_sex = data_popk_year_sex.reset_index()



data_popk_year_sex['deth_rat_100k'] = data_popk_year_sex['total_suicide'] / data_popk_year_sex['total_population'] * 100000

# data_popk_year

plt.figure(figsize=(15, 10))

color = sns.color_palette('husl', n_colors=2)

sns.set(style="darkgrid")

sns.lineplot(x="year", y="deth_rat_100k", hue='sex', marker='o', markersize=10, color='#ec2915', data=data_popk_year_sex, palette=color)



plt.figtext(.5,.91,'Suicide on 100k pop rate by Year and Gender (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_popk_year_age = data_suicide.groupby(by=['year', 'age_range']).agg({'suicides_no':['sum'], 'population': ['sum']})

data_popk_year_age.columns = ['total_suicide', 'total_population']

data_popk_year_age = data_popk_year_age.reset_index()



data_popk_year_age['deth_rat_100k'] = data_popk_year_age['total_suicide'] / data_popk_year_age['total_population'] * 100000

# data_popk_year

plt.figure(figsize=(15, 10))

color = sns.color_palette('Set2', n_colors=6)

sns.set(style="darkgrid")

sns.lineplot(x="year", y="deth_rat_100k", hue='age_range', marker='o', markersize=10, color='#ec2915', data=data_popk_year_age, palette=color)



plt.figtext(.5,.91,'Suicide on 100k pop rate by Year and Age (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_country_total = data_suicide.groupby(by=['country']).agg({'suicides_no': ['sum']})

data_country_total.columns = ['total_suicide']

data_country_total.reset_index(inplace=True)

data_country_total = data_country_total.sort_values(by=['total_suicide'], ascending=False).head(20)

# data_country_total



color = sns.color_palette("Set2")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='total_suicide', y='country', data=data_country_total, palette=color)

plt.xticks(rotation=90)



plt.figtext(.5,.91,'Top 20 countries total suicide (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
fig = px.pie(data_country_total, values='total_suicide', names='country')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
plt.figure(figsize=(20, 15))

countries = ['Russian Federation', 'United States', 'Japan', 'France']

for i, column in enumerate(countries):

    plt.subplot(2, 2, i+1)

    data_rf = data_suicide[data_suicide['country'] == column]

    data_rf_grp = data_rf.groupby(by=['year']).agg({"suicides_no": ['sum']})

    data_rf_grp.columns = ['total_suicide']

    data_rf_grp.reset_index(inplace=True)

    # data_rf_grp = data_rf_grp.sort_values(by=['total_suicide'])

    data_rf_grp



    color = sns.color_palette("Set2")

    sns.set(style="darkgrid")

#     plt.figure(figsize=(15, 10))

    sns.barplot(x='year', y='total_suicide', data=data_rf_grp, palette=color)

    plt.xticks(rotation=90)

    plt.title(f"{column} total suicide by year (1985 - 2015)")

    

plt.figtext(.5,.91,'Top 4 countries total suicide by year (1985 - 2015)', color='b', fontsize=25, ha='center')

data_country_no = data_suicide.groupby(by=['country', 'sex']).agg({'suicides_no': ['sum']})

data_country_no.columns = ['total_suicide']

data_country_no.reset_index(inplace=True)

data_country_name = data_country_no.sort_values(by=['total_suicide'], ascending=False)

data_country_name = data_country_name.head(27)

data_country_name = data_country_name['country']



data_country_no = data_country_no[data_country_no['country'].isin(data_country_name)]

# data_country_no.head()



color = sns.color_palette("Set2")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='country', y='total_suicide', hue='sex', data=data_country_no, palette=color)

plt.xticks(rotation=90)



plt.figtext(.5,.91,'Top 20 countries total suicide Hue Sex (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_map = data_suicide.groupby(by=['country']).agg({"suicides_no": ['sum']})

data_map.columns = ['total_suicide']

data_map.reset_index(inplace=True)

data_map



fig = px.choropleth(data_map, locations="country", locationmode='country names',

                    color="total_suicide", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale='sunset')



fig.update_layout(

    title="Suicide By Country on Map (1985 - 2015)",

    font=dict(

        family="Courier New, monospace",

        size=15,

        color="RebeccaPurple"

    )

)



fig.show()
data_grp_gdp_2 = data_suicide.groupby(by=['country', 'year', 'sex', 'gdp_per_capita ($)']).agg({"suicides_no": ['sum']})

data_grp_gdp_2.columns = ["total_suicide"]

data_grp_gdp_2.reset_index(inplace=True)

# "Russian Federation" "United States" "Japan"



# data_grp_gdp = data_grp_gdp.sort_values(by=['gdp_per_capita ($)'], ascending=False).head(200)

data_grp_gdp_2



data_grp_gdp_cng = data_grp_gdp_2[data_grp_gdp_2['total_suicide'] < 40000]

plt.figure(figsize=(15, 10))

sns.scatterplot(x="gdp_per_capita ($)", y="total_suicide", hue="sex", data=data_grp_gdp_cng)



plt.figtext(.5,.91,'Suicide by GDP per capita (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
plt.figure(figsize=(20, 15))

countries = ['Russian Federation', 'United States', 'Japan', 'France']

for i, column in enumerate(countries):

    plt.subplot(2, 2, i+1)



    data_grp_gdp = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({"suicides_no": ['sum']})

    data_grp_gdp.columns = ["total_suicide"]

    data_grp_gdp.reset_index(inplace=True)

    # data_grp_gdp

    data_grp_gdp_cn = data_grp_gdp[data_grp_gdp['country'] == column]



    sns.scatterplot(x="gdp_per_capita ($)", y="total_suicide", data=data_grp_gdp_cn)

    plt.title(f"{column} total suicide by gdp per capita (1985 - 2015)")

    

plt.figtext(.5,.91,'Top 4 countries total suicide by GDP per capita (1985 - 2015)', color='b', fontsize=25, ha='center')

# plt.show()
test_data = data_suicide.groupby(by=['year']).agg({"gdp_per_capita ($)":['mean'], "suicides_no": ['mean']})

test_data.columns = ["gdp_per_capita", "total_suicide"]

# test_data.reset_index(inplace=True)

test_data



from sklearn import preprocessing

from sklearn.preprocessing import RobustScaler

# min_max_scaller = RobustScaler()

min_max_scaller = preprocessing.MinMaxScaler()

gdp_scale_data = min_max_scaller.fit_transform(test_data)

gdp_scale_data 





suicide_socio_economic_mean_scaled = pd.DataFrame(gdp_scale_data)

suicide_socio_economic_mean_scaled.columns = test_data.columns

suicide_socio_economic_mean_scaled.index = test_data.index

test_data_final = suicide_socio_economic_mean_scaled.reset_index()

test_data_final



plt.figure(figsize=(15, 10))

color = sns.color_palette('husl', n_colors=2)

sns.set(style="darkgrid")

sns.lineplot(x="year", y="gdp_per_capita", data=test_data_final, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#36ada4')

sns.lineplot(x="year", y="total_suicide", data=test_data_final, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#e99675')



plt.figtext(.5,.91,'Year wise total suicide with GDP (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
suicide_socio_economic_mean= data_suicide.pivot_table(['suicides/100k pop','gdp_per_capita ($)'],['year'], aggfunc='mean')

x = suicide_socio_economic_mean.values



from sklearn import preprocessing

min_max_scaller = preprocessing.MinMaxScaler()

gdp_scale_data = min_max_scaller.fit_transform(x)

gdp_scale_data 



suicide_socio_economic_mean_scaled = pd.DataFrame(gdp_scale_data)

suicide_socio_economic_mean_scaled.columns = suicide_socio_economic_mean.columns

suicide_socio_economic_mean_scaled.index = suicide_socio_economic_mean.index

suicide_socio_economic_mean_scaled.reset_index(inplace=True)



plt.figure(figsize=(15, 10))

sns.lineplot(x="year", y="gdp_per_capita ($)", data=suicide_socio_economic_mean_scaled, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#36ada4')

sns.lineplot(x="year", y="suicides/100k pop", data=suicide_socio_economic_mean_scaled, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#e99675')



plt.figtext(.5,.91,'100k population suicide chart with GDP (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_grp_pop = data_suicide.groupby(by=['age_range', 'sex']).agg({"population": ['sum']})

data_grp_pop.columns = ['total_population']

data_grp_pop.reset_index(inplace=True)

# data_grp_pop



color = sns.color_palette("Set2")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='age_range', hue='sex', y='total_population', data=data_grp_pop, palette=color)

plt.xticks(rotation=90)



plt.figtext(.5,.91,'Population by age range (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
plt.figure(figsize=(5, 5))

fig = px.pie(data_grp_pop, values='total_population', names='age_range')

fig.show()
data_grp_pop_sui = data_suicide.groupby(by=['age_range', 'sex']).agg({"population": ['sum'], "suicides_no": ['sum']})

data_grp_pop_sui.columns = ['total_population', "total_suicide"]

data_grp_pop_sui.reset_index(inplace=True)

data_grp_pop_sui



data_grp_pop_sui['deth_rat_100k'] = data_grp_pop_sui['total_suicide'] / data_grp_pop_sui['total_population'] * 100000



color = sns.color_palette("Set2")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='age_range', hue='sex', y='deth_rat_100k', data=data_grp_pop_sui, palette=color)

plt.xticks(rotation=90)



plt.figtext(.5,.91,'Suicide rate 100k Population by age range (1985 - 2015)', color='b', fontsize=25, ha='center')

plt.show()
data_grp_mf = data_suicide.groupby(by=['sex']).agg({"suicides_no": ['sum']})

data_grp_mf.columns = ['total_suicide']

data_grp_mf.reset_index(inplace=True)

# data_grp_mf



fig1, ax1 = plt.subplots(figsize=(15, 7))

wedges, texts, autotexts = ax1.pie(data_grp_mf['total_suicide'], labels=data_grp_mf['sex'], autopct='%1.1f%%', startangle=90, colors=['#646dd5', '#64b1d5'])

ax1.axis('equal')



plt.setp(autotexts, size=15, color="w")

plt.figtext(.5,.91,'Men and Women suicide percentage (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
data_popk_year_sex = data_suicide.groupby(by=['year', 'sex']).agg({'suicides_no':['sum'], 'population': ['sum']})

data_popk_year_sex.columns = ['total_suicide', 'total_population']

data_popk_year_sex = data_popk_year_sex.reset_index()



data_popk_year_sex['deth_rat_100k'] = data_popk_year_sex['total_suicide'] / data_popk_year_sex['total_population'] * 100000

# data_popk_year

plt.figure(figsize=(15, 10))

flatui = ["#e78ac3", "#8da0cb"]

color = sns.color_palette(flatui)

sns.set(style="darkgrid")

sns.lineplot(x="year", y="deth_rat_100k", hue='sex', marker='o', markersize=10, color='#ec2915', data=data_popk_year_sex, palette=color)



plt.figtext(.5,.91,'Commit suicide per 100k people (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
data_sc_pop = data_suicide.groupby(by=['age_range', 'sex']).agg({'suicides_no': ['sum'], 'population': ['sum']})

data_sc_pop.columns = ['total_suicide', 'total_population']

data_sc_pop.reset_index(inplace=True)



data_sc_pop['deth_rat_100k'] = data_sc_pop['total_suicide'] / data_sc_pop['total_population'] * 100000

data_sc_pop



flatui = ["#e78ac3", "#8da0cb"]

color = sns.color_palette(flatui)

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='age_range', y='deth_rat_100k', hue='sex', data=data_sc_pop, palette=color)



plt.setp(autotexts, size=10, color="w")

plt.figtext(.5,.91,'Suicide rate in 100k population by age (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
data_sc_no = data_suicide.groupby(by=['age_range', 'sex']).agg({'suicides_no': ['sum']})

data_sc_no.columns = ['total_suicide']

data_sc_no.reset_index(inplace=True)

# data_sc_no



flatui = ["#ffd92f", "#a6d854"]

color = sns.color_palette(flatui)

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='age_range', y='total_suicide', hue='sex', data=data_sc_no, palette=color)



plt.setp(autotexts, size=10, color="w")

plt.figtext(.5,.91,'Total suicide by age range (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
suicide_socio_economic_mean= data_suicide.pivot_table(['suicides/100k pop','gdp_per_capita ($)'],['year'], aggfunc='mean')

x = suicide_socio_economic_mean.values



from sklearn import preprocessing

min_max_scaller = preprocessing.MinMaxScaler()

gdp_scale_data = min_max_scaller.fit_transform(x)

gdp_scale_data 



suicide_socio_economic_mean_scaled = pd.DataFrame(gdp_scale_data)

suicide_socio_economic_mean_scaled.columns = suicide_socio_economic_mean.columns

suicide_socio_economic_mean_scaled.index = suicide_socio_economic_mean.index

suicide_socio_economic_mean_scaled.reset_index(inplace=True)



plt.figure(figsize=(15, 10))

sns.lineplot(x="year", y="gdp_per_capita ($)", data=suicide_socio_economic_mean_scaled, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#36ada4')

sns.lineplot(x="year", y="suicides/100k pop", data=suicide_socio_economic_mean_scaled, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#e99675')



plt.figtext(.5,.91,'Suicide rate for every 100k people with GDP by year (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
len(data_suicide['country'].unique())
sri_data = data_suicide[data_suicide['country'] == 'United States']

len(sri_data['year'].unique())
year_count_data = data_suicide.groupby(by=['country']).agg({'year':['nunique']})

year_count_data.columns = ['year_count']

year_count_data.reset_index(inplace=True)

# year_count_data
plt.figure(figsize=(15, 10))

data_grp_gdp = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({'suicides_no': ['sum'], 'population': ['sum']})

data_grp_gdp.columns = ['total_suicide', 'total_population']

data_grp_gdp.reset_index(inplace=True)



data_grp_gdp['deth_rat_100k'] = data_grp_gdp['total_suicide'] / data_grp_gdp['total_population'] * 100000

data_grp_gdp



data_grp_gdp['prev_value'] = data_grp_gdp.groupby('country')['deth_rat_100k'].shift(periods=10)





data_grp_gdp = data_grp_gdp.groupby(by=['country']).tail(10)

data_grp_gdp = data_grp_gdp.groupby(by=['country']).agg({'deth_rat_100k': ['sum'], 'prev_value': ['sum']})

data_grp_gdp.columns = ['total_deth', 'total_prev']

data_grp_gdp.reset_index(inplace=True)







data_grp_gdp['diff'] = data_grp_gdp['total_deth'] - data_grp_gdp['total_prev']

data_grp_gdp = data_grp_gdp[data_grp_gdp['diff'] > 0]

data_grp_gdp = data_grp_gdp.sort_values(by=['diff'], ascending=False)

data_grp_gdp = pd.merge(data_grp_gdp, year_count_data, on=['country'], how='left')



total_country = data_grp_gdp



data_grp_gdp = data_grp_gdp[data_grp_gdp['year_count'] > 30]

data_grp_gdp = data_grp_gdp.head(4)

country_names = data_grp_gdp['country'].values

country_names

total_country = total_country['country'].values

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(total_country))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
plt.figure(figsize=(20, 15))

for i, column in enumerate(country_names):

    plt.subplot(2, 2, i+1)



    data_grp_gdp = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({'suicides_no': ['sum'], 'population': ['sum']})

    data_grp_gdp.columns = ['total_suicide', 'total_population']

    data_grp_gdp.reset_index(inplace=True)



    data_grp_gdp['deth_rat_100k'] = data_grp_gdp['total_suicide'] / data_grp_gdp['total_population'] * 100000

#     data_grp_gdp

    # data_grp_gdp

    data_grp_gdp_cn = data_grp_gdp[data_grp_gdp['country'] == column]



    sns.lineplot(x="gdp_per_capita ($)", y="deth_rat_100k", data=data_grp_gdp_cn, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#e99675')

    plt.title(f"{column} per 100k suicide by gdp per capita (1985 - 2015)")

    

plt.figtext(.5,.91,'Top 4 countries total suicide by GDP per capita (1985 - 2015)', color='b', fontsize=25, ha='center')

# plt.show()
plt.figure(figsize=(20, 15))

countries = ['Suriname', 'Guyana', 'Republic of Korea', 'Montenegro']

for i, column in enumerate(country_names):

    plt.subplot(2, 2, i+1)

    data_grp_gdp = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({"suicides_no": ['sum']})

    data_grp_gdp.columns = ["total_suicide"]

    data_grp_gdp.reset_index(inplace=True)

    # data_grp_gdp

    data_grp_gdp_cn = data_grp_gdp[data_grp_gdp['country'] == column]



    sns.lineplot(x="gdp_per_capita ($)", y="total_suicide", data=data_grp_gdp_cn, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#e99675')

    plt.title(f"{column} total suicide by gdp per capita (1985 - 2015)")

#     plt.title(f"United States total suicide by gdp per capita (1985 - 2015)")
map_gen = {'25-34': 'Boomers', '35-54': 'Silent', '15-24': 'Generation X', '55-74': 'Millenials', '75+': 'G.I. Generation', '5-14': 'Generation Z'}
copy_data = data_suicide.copy()

copy_data['generation'] = copy_data['age_range'].map(map_gen)

# copy_data
data_sc_no = copy_data.groupby(by=['generation']).agg({'suicides_no': ['sum']})

data_sc_no.columns = ['total_suicide']

data_sc_no.reset_index(inplace=True)

data_sc_no = data_sc_no.sort_values(by=['total_suicide'], ascending=False)

# data_sc_no



color = sns.color_palette("coolwarm")

sns.set(style="darkgrid")

plt.figure(figsize=(15, 10))

sns.barplot(x='total_suicide', y='generation', data=data_sc_no, palette=color)



plt.setp(autotexts, size=10, color="w")

plt.figtext(.5,.91,'Worldwide Suicide on total suicide by generation (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
data_popk_year_age = copy_data.groupby(by=['year', 'generation']).agg({'suicides_no':['sum'], 'population': ['sum']})

data_popk_year_age.columns = ['total_suicide', 'total_population']

data_popk_year_age = data_popk_year_age.reset_index()



data_popk_year_age['deth_rat_100k'] = data_popk_year_age['total_suicide'] / data_popk_year_age['total_population'] * 100000

# data_popk_year

plt.figure(figsize=(15, 10))

color = sns.color_palette('coolwarm', n_colors=6)

sns.set(style="darkgrid")

# sns.barplot(x='generation', y='deth_rat_100k', data=data_popk_year_age, palette=color)

sns.lineplot(x="year", y="deth_rat_100k", hue='generation', marker='o', markersize=10, color='#ec2915', data=data_popk_year_age, palette=color)



plt.figtext(.5,.91,'Suicide on 100k pop rate by Year and generation (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
plt.figure(figsize=(15, 10))

data_grp_gdp = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({'suicides_no': ['sum'], 'population': ['sum']})

data_grp_gdp.columns = ['total_suicide', 'total_population']

data_grp_gdp.reset_index(inplace=True)



data_grp_gdp['deth_rat_100k'] = data_grp_gdp['total_suicide'] / data_grp_gdp['total_population'] * 100000

data_grp_gdp



data_grp_gdp['prev_value'] = data_grp_gdp.groupby('country')['deth_rat_100k'].shift(periods=10)





data_grp_gdp = data_grp_gdp.groupby(by=['country']).tail(10)

data_grp_gdp = data_grp_gdp.groupby(by=['country']).agg({'deth_rat_100k': ['sum'], 'prev_value': ['sum']})

data_grp_gdp.columns = ['total_deth', 'total_prev']

data_grp_gdp.reset_index(inplace=True)







data_grp_gdp['diff'] = data_grp_gdp['total_deth'] - data_grp_gdp['total_prev']

data_grp_gdp = data_grp_gdp[data_grp_gdp['diff'] > 0]

data_grp_gdp = data_grp_gdp.sort_values(by=['diff'], ascending=False)

data_grp_gdp = pd.merge(data_grp_gdp, year_count_data, on=['country'], how='left')



total_country = data_grp_gdp



data_grp_gdp = data_grp_gdp[data_grp_gdp['year_count'] > 30]

data_grp_gdp = data_grp_gdp.head(4)

country_names = data_grp_gdp['country'].values

country_names
plt.figure(figsize=(15, 10))

data_trend = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({'suicides_no': ['sum'], 'population': ['sum']})

data_trend.columns = ['total_suicide', 'total_population']

data_trend.reset_index(inplace=True)

data_trend = pd.merge(data_trend, year_count_data, on=['country'], how='left')

data_trend = data_trend[data_trend['year_count'] > 30]



data_trend['deth_rat_100k'] = data_trend['total_suicide'] / data_trend['total_population'] * 100000



data_trend['prev_value_1'] = data_trend.groupby('country')['deth_rat_100k'].shift(periods=1)

data_trend['diff_1'] = data_trend['deth_rat_100k'] - data_trend['prev_value_1']



data_trend['prev_value_2'] = data_trend.groupby('country')['deth_rat_100k'].shift(periods=2)

data_trend['diff_2'] = data_trend['deth_rat_100k'] - data_trend['prev_value_2']

# data_trend
us_data_trend = data_trend[data_trend['country'] == 'United States']

# us_data_trend
def change_colour(val):

   return ['background-color: #a0f0a0' if x < 0  else 'background-color: #f9c1c1' for x in val]

us_data_trend_1 = us_data_trend.copy()

us_data_trend_1 = us_data_trend_1[['country', 'year', 'deth_rat_100k', 'prev_value_1', 'diff_1', 'prev_value_2', 'diff_2']]

us_data_trend_1 = us_data_trend_1.style.apply(change_colour, axis=1, subset=['diff_1', 'diff_2'])

us_data_trend_1
plt.figure(figsize=(15, 10))

sns.lineplot(x="year", y="diff_1", data=us_data_trend, palette="tab10", marker='o', markersize=10, linewidth=2.5, color='#e99675')

plt.figtext(.5,.91,'United States suicide trend lag-1 (1985 - 2015)', color='#ee4a09', fontsize=25, ha='center')

plt.show()
sns.set(style="darkgrid")

sns.jointplot("population", "suicides_no", data=data_suicide, kind="reg", color="#e99675", height=10)

plt.figtext(.5,1.05,'Population wise suicide with suicides no (1985 - 2015)', color='#6788ee', fontsize=25, ha='center')

plt.show()
data_grp_gdp = data_suicide.groupby(by=['country', 'year', 'gdp_per_capita ($)']).agg({'suicides_no': ['sum'], 'population': ['sum']})

data_grp_gdp.columns = ['total_suicide', 'total_population']

data_grp_gdp.reset_index(inplace=True)



data_grp_gdp['deth_rat_100k'] = data_grp_gdp['total_suicide'] / data_grp_gdp['total_population'] * 100000

data_grp_gdp

total_country = data_grp_gdp

total_country = total_country.sort_values(by=['deth_rat_100k'], ascending=False)

total_country = total_country['country'].values

# total_country

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='#000',

                          width=650,

                          height=550,

                         ).generate(" ".join(total_country))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.figtext(.5,.91,'Top countries in 100k death rate', color='#6788ee', fontsize=25, ha='center')

plt.show()
from mpl_toolkits.mplot3d import axes3d

suicide_3d = data_suicide.copy()



sns.set(rc={'figure.figsize':(15,10)})

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(suicide_3d['gdp_per_capita ($)'], suicide_3d['year'], suicide_3d['suicides/100k pop'], alpha=0.2, c="red", s=30) 

plt.title('gdp_per_capita ($), year, suicides/100k pop')

plt.legend(loc=2)

plt.figtext(.5,.91,'3d plot with gdp_per_capita, year, suicides/100k pop', color='#6788ee', fontsize=25, ha='center')

plt.show()