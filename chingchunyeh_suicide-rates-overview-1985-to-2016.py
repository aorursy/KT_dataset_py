import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np 

import os 

import pandas as pd

from scipy import stats

import pycountry

import geopandas
print(os.listdir('../input'))
filename = "../input/master.csv"

df = pd.read_csv(filename)

print(df.head())
df.rename(columns={"suicides/100k pop":"suicides_pop","HDI for year":"HDI_for_year",

                  " gdp_for_year ($) ":"gdp_for_year"," gdp_per_capita ($) ":"gdp_per_capita",

                    "gdp_per_capita ($)":"gdp_per_capita"}, inplace=True)

print(df.columns)
df["gdp_for_year"] = df["gdp_for_year"].str.replace(",","").astype(np.int64)

df["age"] = df["age"].str.replace("5-14 years","05-14 years")
df_men = df[df.sex == "male"]

df_women = df[df.sex == "female"]

sns.lineplot(df_men.year, df.suicides_no, ci = None)

sns.lineplot(df_women.year, df.suicides_no, ci = None)

plt.legend(["male", 'female'])

plt.show()
df_age = df.groupby(["year","age"])["suicides_no", "population"].sum()

df_reset = df_age.copy().reset_index()

plt.figure(figsize=(9,6))

sns.lineplot("year", df_reset.suicides_no*100/df_reset.population, hue = "age",

             data = df_reset, linewidth = 2.5, style = "age", markers=True

            , dashes=False)

plt.xticks(rotation = 90)

plt.show()
df_generation = df.groupby(["year", "generation"])["suicides_no", "population"].sum()

df_generation_reset = df_generation.copy().reset_index()

plt.figure(figsize=(9,6))

sns.lineplot("year", df_generation_reset.suicides_no*100/df_generation_reset.population, hue = "generation", 

            data = df_generation_reset, linewidth = 2.5, style = "generation", markers=True

            , dashes=False)

plt.xticks(rotation = 90)

plt.show()
df1 = df.groupby("country")["suicides_no"].sum()

country_name = list(df1.index.get_level_values(0))

len(country_name)
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3
country_not_in_list = [i for i in country_name[:] if i not in countries.keys()]

country_not_in_list
df.replace("Republic of Korea", "Korea, Republic of", inplace = True)

df.replace('Czech Republic', "Czechia", inplace = True)

df.replace('Macau', 'Macao', inplace = True)

df.replace('Saint Vincent and Grenadines', "Saint Vincent and the Grenadines", inplace = True)
df_suino = df.groupby(["country","year"])["suicides_no"].sum()

df_sum = df_suino.sort_index(ascending=True)[:] * 100



df_pop = df.groupby(["country","year"]).population.sum()

df_pop_sum = df_pop.sort_index(ascending=False)[:]



df_total = df_sum / df_pop_sum

df_total.head(10)
country_dict={}

for country in df_total.index.get_level_values(0):

    if country not in country_dict.keys():

        country_dict[country] = df_total[country].mean()

    else:

        pass



tup = list(country_dict.items())

tup.sort(key= lambda pair:pair[1], reverse = True)



country_list = [a[0] for a in tup]

country_suicide = [a[1] for a in tup]
plt.figure(figsize=(8,32))

sns.barplot(x=country_suicide[:],y=country_list[:], palette="GnBu")

plt.xlabel("ratio of suicide")

plt.ylabel("country")

plt.title("suicide rate vs country")

plt.show()
country_dict = dict()

for idx in range(len(country_list)):

    country_dict[countries[country_list[idx]]] = country_suicide[idx]
new_country_dict = {}

new_country_dict["iso_a3"] = list(country_dict.keys())

new_country_dict["suicide_rate"] = list(country_dict.values())

new_country_df = pd.DataFrame(new_country_dict)

new_country_df.head()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))

world.head()
result = pd.merge(world, new_country_df, on = "iso_a3")

result.head()
ax = result.plot()

ax.set_title("world suicide rate")

result.plot(column='suicide_rate', ax = ax, legend=True)
plt.figure(figsize = (9,6))

for country in country_list[:10]:

    plt.plot(df_total[country].index,df_total[country].values, label=country, marker="o")

plt.xlabel("year")

plt.ylabel("ratio of suicide")

plt.legend()

plt.show()
plt.figure(figsize = (9,6))

df_gdp = df.groupby(["country","year"]).gdp_per_capita.mean()

for country in country_list[:10]:

    plt.plot(df_gdp[country].index,df_gdp[country].values, label=country, marker="o")

plt.xlabel("year")

plt.ylabel("gdp_per_capita")

plt.legend()

plt.show()
plt.figure(figsize = (9,6))

for country in country_list[:10]:

    sns.regplot(x=df_gdp[country].values, y=df_total[country].values, label = country)

plt.xlabel("gdp_per_capita")

plt.ylabel("ratio of suicides")

plt.ylim(0,0.06)

plt.xlim(0)

plt.legend()

plt.show()



corr_eff = {}

for country in country_list[:10]:

    slope, intercept, r_value, p_value, std_err = stats.linregress(df_gdp[country].values,df_total[country].values)

    corr_eff[country] = float(r_value)

    

sns.barplot(x=list(corr_eff.keys()), y=list(corr_eff.values()), palette = "YlOrRd")

plt.xticks(rotation = 90)

plt.xlabel("Country")

plt.ylabel("correlation coeff.")

plt.title("GDP vs suicides")

plt.show()
corr_eff = {}

p_value_eff = {}

for country in country_list[:]:

    slope, intercept, r_value, p_value, std_err = stats.linregress(df_gdp[country].values, df_total[country].values)

    corr_eff[country] = float(r_value)

    p_value_eff[country] = float(p_value)



gdp_tup = list(corr_eff.items())

gdp_tup.sort(key= lambda pair:pair[1], reverse = False)

dgp_relation = {a[0]:a[1] for a in gdp_tup}



plt.figure(figsize=(18,12))

sns.barplot(x=list(dgp_relation.keys()), y=list(dgp_relation.values()), palette = "YlOrRd")

plt.xticks(rotation = 90)

plt.xlabel("Country")

plt.ylabel("correlation coeff.")

plt.title("GDP vs suicides")

plt.show()
high_relation_gdp = {a:b for a,b in dgp_relation.items() if b <= -0.6}

print(len(high_relation_gdp))

high_relation_gdp
positive_relation_gdp = {a:b for a,b in corr_eff.items() if b > 0.6}

positive_relation_tup = list(positive_relation_gdp.items())

positive_relation_tup.sort(key= lambda pair:pair[1], reverse = True)

positive_relation = {a[0]:a[1] for a in positive_relation_tup}

print(len(positive_relation))

positive_relation
city_list = list({a:b for a,b in positive_relation.items()})

for country in city_list[:10]:

    plt.plot(df_gdp[country].index,df_gdp[country].values, label=country, marker="o")

plt.xlabel("year")

plt.ylabel("gdp_per_capita")

plt.legend()

plt.show()



for country in city_list[:10]:

    plt.plot(df_total[country].index,df_total[country].values, label=country, marker="o")

plt.xlabel("year")

plt.ylabel("ratio of suicide")

plt.legend()

plt.show()
plt.figure(figsize = (9,6))

sns.barplot(x="year", y="suicides_pop", hue="age",

            ci = None,data = (df[df["country"] == "Korea, Republic of"]) )

plt.xticks(rotation = 90)

plt.title("suicide rate of Korea" )

plt.legend()    

plt.show()
without_relation_gdp = {a:b for a,b in corr_eff.items() if -0.3 < b < 0.3}

no_relation_gdp = [i for i in country_list[:20] if i in without_relation_gdp.keys()]

no_relation_gdp
plt.figure(figsize = (9,6))

sns.barplot(x="year", y="suicides_pop", hue="age",

            ci = None,data = (df[df["country"] == "Japan"]) )

plt.xticks(rotation = 90)

plt.title("suicide rate of Japan" )

plt.legend()    

plt.show()