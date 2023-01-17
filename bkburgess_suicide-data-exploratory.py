# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
location = "../input/suicide-rates-overview-1985-to-2016/master.csv"

suicides = pd.read_csv(location)



print("FIRST FIVE ENTRIES:")

print(suicides.head())



print("\n\n\n\n\n\nBASIC DESCRIPTIVE STATISTICS:")

print(suicides.describe())



print("\n\n\n\n\n\nNUMBER OF ENTRIES:")

print(suicides.info())

print(suicides.columns)
######## FIXING gdp_for_year ##########



suicides = suicides.rename(columns ={" gdp_for_year ($) ": "gdp_for_year ($)"})

suicides["gdp_for_year ($)"] = suicides["gdp_for_year ($)"].apply(lambda x: int(x.replace(',' , '')))



print("FIRST FIVE ENTRIES:")

print(suicides.head())



print("\n\n\n\n\n\nBASIC DESCRIPTIVE STATISTICS:")

print(suicides.describe())



print("\n\n\n\n\n\nNUMBER OF ENTRIES:")

print(suicides.info())
alb1987 = suicides[[i&j for i,j in zip(suicides["country"] == "Albania", suicides["year"] == 1987)]]



print(sum(alb1987["population"]))

print(alb1987)
fig, (ax_gdp_pc, ax_gdp_yr, ax_sn, ax_sr, ax_pop) = plt.subplots(figsize=(20,10), nrows=5)

sns.set(style="darkgrid")

ln=True



sns.distplot(a=suicides["suicides_no"], ax=ax_sn, hist_kws = {"log":ln})

sns.distplot(a=suicides["suicides/100k pop"], ax=ax_sr, hist_kws = {"log":ln})

sns.distplot(a=suicides["population"], ax=ax_pop, hist_kws = {"log":ln})

sns.distplot(a=suicides["gdp_for_year ($)"], ax=ax_gdp_yr, hist_kws = {"log":ln})

sns.distplot(a=suicides["gdp_per_capita ($)"], ax=ax_gdp_pc, hist_kws = {"log":ln})

plt.title("")

plt.xticks(rotation=45)
for col in suicides.columns:

    print(f"{col}: {len(suicides[col].unique())}")
missing_hdi = suicides[suicides["HDI for year"].isnull()]

print("MISSING HDI HEAD:")

print(missing_hdi.head())



print("\n\n\n\nBY COUNTRY:")

print(len(missing_hdi["country"].unique()))



print("\n\n\n\nBY YEAR")

print(len(missing_hdi["year"].unique()))



print("\n\n\n\nBY COUNTRY-YEAR")

print(len(missing_hdi["country-year"].unique()))
b = []

for yr in pd.unique(suicides["year"]):

    a = suicides[suicides["year"] == yr]

    #print(a["HDI for year"])

    b.append(len(a["HDI for year"].dropna())/12)

p = sns.barplot(x=pd.unique(suicides["year"]), y=b)

plt.xticks(rotation=60)

plt.show()
b = []

for ct in pd.unique(suicides["country"]):

    a = suicides[suicides["country"] == ct]

    #print(a["HDI for year"])

    b.append(len(a["HDI for year"].dropna())/12)



fig, ax = plt.subplots(figsize=(23,6))

ax = sns.barplot(x=pd.unique(suicides["country"]), y=b)

plt.xticks(rotation=60)

fig.show()
p = sns.lineplot(x="year", y="HDI for year", data=suicides, hue="country")

p.legend_.remove()

plt.show()
rate = suicides.groupby(by=["country"]).mean()

rate = rate.sort_values(by=["suicides/100k pop"], ascending=False)

fig = plt.figure(figsize=(20,17))

sns.barplot(y=rate.index, x=rate["suicides/100k pop"])

plt.title("suicide rate by country")

fig.show()

#plt.xticks(rotation=45)
time = suicides.groupby(by="year").mean()

plt.figure(figsize=(10,5))

sns.barplot(x=time.index, y=time["suicides/100k pop"])

plt.title("showing the relation between year and no of suicides cases")

plt.xticks(rotation=45)
# choose the years to compare

after_year = 2016

before_year = 2015

    

# set up data sets to compare

recent = suicides[suicides["year"] == after_year]

recent = recent.groupby(by="country").sum()

recent = recent.sort_values(by=["suicides/100k pop"], ascending=False)



old = suicides[suicides["year"] == before_year]

old = old.groupby(by="country").sum()

old = old.sort_values(by=["suicides/100k pop"], ascending=False)





# create plot of older and recent data sets

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize = (30,20))

sns.barplot(y=old.index, x=old["suicides/100k pop"], ax = ax1)

sns.barplot(y=recent.index, x=recent["suicides/100k pop"], ax = ax2)



# make sure they use the same scale for easy comparison

mx = max(max(recent["suicides/100k pop"]), max(old["suicides/100k pop"]))

ax1.set_xlim(0,1.1*mx)

ax2.set_xlim(0,1.1*mx)



ax1.set_title(f"Before {before_year}")

ax2.set_title(f"After {after_year}")



ax1.invert_xaxis()

ax1.yaxis.tick_right()

ax2.yaxis.tick_left()

fig.show()
# choose the years to compare

after_year = 2016

before_year = 2015

    

# set up data sets to compare

recent = suicides[suicides["year"] == after_year]

weighted_rate = pd.Series(index=recent.index, name="weighted_suicide_rate")

for country in pd.unique(recent["country"]):

    idx = recent[recent["country"] == country].index

    weighted_rate.loc[idx] = recent.loc[idx, "suicides_no"]*100_000/sum(recent.loc[idx, "population"])

recent = recent.join(weighted_rate)

recent = recent.groupby(by="country").sum()

recent = recent.sort_values(by=["weighted_suicide_rate"], ascending=False)



old = suicides[suicides["year"] == before_year]

weighted_rate = pd.Series(index=old.index, name="weighted_suicide_rate")

for country in pd.unique(old["country"]):

    idx = old[old["country"] == country].index

    weighted_rate.loc[idx] = old.loc[idx, "suicides_no"]*100_000/sum(old.loc[idx, "population"])

old = old.join(weighted_rate)

old = old.groupby(by="country").sum()

old = old.sort_values(by=["weighted_suicide_rate"], ascending=False)





# create plot of older and recent data sets

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize = (30,20))

sns.barplot(y=old.index, x=old["weighted_suicide_rate"], ax = ax1)

sns.barplot(y=recent.index, x=recent["weighted_suicide_rate"], ax = ax2)



# make sure they use the same scale for easy comparison

mx = max(max(recent["weighted_suicide_rate"]), max(old["weighted_suicide_rate"]))

ax1.set_xlim(0,1.1*mx)

ax2.set_xlim(0,1.1*mx)



ax1.set_title(f"Before {before_year}")

ax2.set_title(f"After {after_year}")



ax1.invert_xaxis()

ax1.yaxis.tick_right()

ax2.yaxis.tick_left()

fig.show()
countries = [

    "United States",

    #"Germany",

    #"France",

    #"United Kingdom",

    #"Canada",

    #"Japan",

    #"Singapore",

    #"Denmark",

    #"Sweden",

    #"Norway",

    #"Poland",

    #"Russian Federation",

    #"Finland",

    #"Uruguay",

    #"Guatemala",

    "Lithuania",

    "Republic of Korea"

]

#print(suicides["country"].apply(lambda x: x in countries))

country_time = suicides[suicides["country"].apply(lambda x: x in countries)].groupby(by=["country","year"]).mean()

country_time = country_time.reset_index(level = "country")



plt.figure(figsize=(10,5))



sns.lineplot(x=country_time.index, y=country_time["suicides/100k pop"], hue = country_time["country"])



plt.title("showing the relation between year and no of suicides cases")

plt.xticks(rotation=45)
age1 = suicides.groupby(by="age").mean()



age2 = suicides[suicides["year"] >= 2011]

age2 = age2.groupby(by="age").mean()



age3 = suicides[suicides["year"] <= 1990]

age3 = age3.groupby(by="age").mean()



fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize =(20,5))

sns.barplot(x=age1.index, y=age1["suicides/100k pop"], ax=ax1)

sns.barplot(x=age2.index, y=age2["suicides/100k pop"], ax=ax2)

sns.barplot(x=age3.index, y=age3["suicides/100k pop"], ax=ax3)



ax1.tick_params(axis='x', labelrotation=45)

ax2.tick_params(axis='x', labelrotation=45)

ax3.tick_params(axis='x', labelrotation=45)



ax1.set_title("Global, all years")

ax2.set_title("Global, before 1990")

ax3.set_title("Global, after 2011")
age_us = suicides[suicides["country"] == "United States"]

age_increasing_us1 = age_us[age_us["year"] <= 2006]

age_increasing_us2 = age_us[age_us["year"] >= 2006]



age_increasing_us1 = age_increasing_us1.groupby(by="age").mean()

age_increasing_us2 = age_increasing_us2.groupby(by="age").mean()



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))

sns.barplot(x=age_increasing_us1.index, y=age_increasing_us1["suicides/100k pop"], ax=ax1)

sns.barplot(x=age_increasing_us2.index, y=age_increasing_us2["suicides/100k pop"], ax=ax2)



ax1.set_title("age of suicides in US before 2006")

ax2.set_title("age of suicides in US after 2006")



mx = max(max(age_increasing_us1["suicides/100k pop"]), max(age_increasing_us2["suicides/100k pop"]))

ax1.set_ylim(ymax=1.1*mx)

ax2.set_ylim(ymax=1.1*mx)



ax1.tick_params(axis='x', labelrotation=45)

ax2.tick_params(axis='x', labelrotation=45)
male = suicides[suicides["sex"] >= "male"]

male = male.groupby(by="country").mean()

male = male.sort_values(by=["suicides/100k pop"], ascending=False)



female = suicides[suicides["sex"] <= "female"]

female = female.groupby(by="country").mean()

female = female.sort_values(by=["suicides/100k pop"], ascending=False)





fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize = (30,20))

sns.barplot(y=female.index, x=female["suicides/100k pop"], ax = ax1)

sns.barplot(y=male.index, x=male["suicides/100k pop"], ax = ax2)



mx = max(max(male["suicides/100k pop"]), max(female["suicides/100k pop"]))

ax1.set_xlim(0,1.1*mx)

ax2.set_xlim(0,1.1*mx)



ax2.set_title("MALE")

ax1.set_title("FEMALE")



ax1.invert_xaxis()

ax1.yaxis.tick_right()

ax2.yaxis.tick_left()

fig.show()
male = suicides[suicides["sex"] >= "male"]

male = male.groupby(by="year").mean()

male = male.sort_values(by=["suicides/100k pop"], ascending=False)



female = suicides[suicides["sex"] <= "female"]

female = female.groupby(by="year").mean()

female = female.sort_values(by=["suicides/100k pop"], ascending=False)





fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True, figsize = (30,20))

sns.barplot(x=male.index, y=male["suicides/100k pop"], ax =ax1)

sns.barplot(x=female.index, y=female["suicides/100k pop"], ax =ax2)



mx = max(max(male["suicides/100k pop"]), max(female["suicides/100k pop"]))

ax1.set_ylim(0, 1.1*mx)

ax2.set_ylim(0, 1.1*mx)



ax1.set_title("MALE")

ax2.set_title("FEMALE")





fig.show()
gdp = suicides



plt.figure(figsize=(10,5))

sns.scatterplot(x=np.log(gdp["gdp_for_year ($)"]), y=gdp["suicides/100k pop"])

plt.title("showing the relation between gdp and no of suicides cases")

plt.xticks(rotation=45)
gdp = suicides



plt.figure(figsize=(10,5))

sns.scatterplot(x=gdp["gdp_per_capita ($)"], y=gdp["suicides/100k pop"])

plt.title("showing the relation between gdp and no of suicides cases")

plt.xticks(rotation=45)
gdp = suicides.groupby(by="country-year").mean()



plt.figure(figsize=(10,5))

sns.distplot(a=gdp["gdp_for_year ($)"])

plt.title("showing the distribution of gdp for countries and years")

plt.xticks(rotation=45)
gdp = suicides.groupby(by="country-year").mean()



plt.figure(figsize=(10,5))

sns.distplot(a=np.log(gdp["gdp_for_year ($)"]))

plt.title("showing the distribution of gdp for countries and years")
plt.figure(figsize=(10,5))

sns.distplot(a=gdp["gdp_per_capita ($)"])

plt.title("showing the distribution of gdp per capita")

plt.xticks(rotation=45)
pop = suicides.groupby(by="country-year").mean()



plt.figure(figsize=(10,5))

sns.scatterplot(x=pop["population"], y=pop["suicides/100k pop"])

plt.title("showing the relation between population and no of suicides cases")

plt.xticks(rotation=45)