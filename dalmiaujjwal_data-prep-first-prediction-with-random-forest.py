import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder

from sklearn.cluster import KMeans

import datetime as dtm

import missingno

import holidays

from collections import Counter
raw_data1 = pd.read_csv("../input/train.csv")

raw_data2 = pd.read_csv("../input/test.csv")

raw_data2["revenue"] = 0

raw_data1["train_test"] = "train"

raw_data2["train_test"] = "test"

raw_data = pd.concat([raw_data1, raw_data2],ignore_index=True)

input_Data = pd.DataFrame(raw_data)

input_Data["release_date_mod"] = pd.to_datetime(input_Data["release_date"], format="%m/%d/%y")

input_Data["identifier"] = input_Data["id"]

input_Data = input_Data.drop(["id","homepage", "imdb_id", "original_title", "overview", "poster_path", "tagline", "title"],axis = 1)
#### Columns with dictionaries are originally read as string. Converting them in to dictionary

columns_with_dictionaries = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

for cols in columns_with_dictionaries:

    input_Data[cols] = input_Data[cols].apply(lambda x: {} if pd.isna(x) else eval(x))
missingno.bar(raw_data1,figsize=(15,6))
missingno.bar(raw_data2,figsize=(15,6))
#### Indicator if the movie belongs to a certain collection or series



input_Data["collection"] = input_Data["belongs_to_collection"].apply(lambda x: 0 if len(x) == 0 else 1)

input_Data = input_Data.drop("belongs_to_collection", axis = 1)
#### Checking if movies belonging to collection have higher average revenue as compared to non collection movies. 

#### The difference is clearly visible



sns.barplot(x = "collection", y = "revenue",estimator=np.mean, data=input_Data)

plt.show()
input_Data["genre_extract"] = input_Data["genres"].apply(lambda x: ["genre na"] if len(x) == 0 else [i["name"] for i in x])

input_Data["genre_count"] = input_Data["genre_extract"].apply(lambda x: 0 if "genre na" in x else len(x))
plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

sns.countplot("genre_count",data=input_Data)

plt.subplot(1,2,2)

sns.barplot(x = "genre_count",y = "revenue",data=input_Data)

plt.show()
#### Converting the list of values to columns

mlb = MultiLabelBinarizer()

input_Data = input_Data.join(pd.DataFrame(mlb.fit_transform(input_Data["genre_extract"]),columns=mlb.classes_, index=input_Data.index))
#### deleting original variables which are not required

input_Data = input_Data.drop(["genres","genre_extract"], axis = 1)
sum(input_Data["release_date_mod"].isnull())
#### There was 1 movie with no release date. Googled and filled the actual release date.

dtval = pd.to_datetime('01-05-2005', format= '%d-%m-%Y')

input_Data.loc[3828,"release_date_mod"] = dtval 
#### Generating other variables using Date

input_Data["year"] = input_Data["release_date_mod"].dt.year

input_Data["month"] = input_Data["release_date_mod"].dt.month

input_Data["day"] = input_Data["release_date_mod"].dt.day

input_Data["year"] = input_Data["year"].apply(lambda x: x-100 if x>2019 else x)

input_Data["release_date_mod"] = pd.to_datetime(input_Data[['year','month','day']])

input_Data["day_of_week"] = input_Data["release_date_mod"].dt.weekday_name

input_Data["week_of_year"] = input_Data["release_date_mod"].dt.weekofyear
input_Data = pd.concat([input_Data,pd.get_dummies(input_Data["day_of_week"])],axis = 1)

input_Data = input_Data.drop("day_of_week",axis=1)
#### Understanding if the release date was a holiday

input_Data["holiday1"] = 0

Countries = ['AR','AU','AT','BY','BE','BR','BG','CA','CO','HR','CZ','DK','FI','FRA','DE','HU','IND','IE','IT','JP','LT','LU','MX','NL','NZ','NO','PL','PT','PTE','RU','SI','SK','ZA','ES','SE','CH','UA','UK','US']

for i in range(len(Countries)):

    for j in range(len(input_Data)):

        try:

            if input_Data.loc[j,"holiday1"] == 1:

                continue

            elif input_Data.loc[j,"release_date_mod"] in holidays.CountryHoliday(Countries[i]):

                input_Data.loc[j,"holiday1"] = 1

                continue

            else:

                continue

        except:

            continue
input_Data = input_Data.drop(["release_date", "release_date_mod"], axis = 1)
budget = pd.DataFrame(input_Data[["identifier","year","budget"]])
cluster = budget["budget"].values.reshape(-1,1)
score = []

for i in range(1,20):

    model = KMeans(n_clusters=i)

    model.fit(cluster)

    score.append(model.inertia_)
plt.plot(score,marker = ".")

plt.xticks(range(0,19,1))

plt.annotate("# of clusters", xy =(3,1000000000000000000), xytext = (7.5,5000000000000000000), arrowprops = dict(facecolor = "black"))

plt.show()
check = pd.pivot_table(budget, index="year", values="budget", aggfunc="mean").reset_index()
plt.figure(figsize=(15,6))

sns.barplot(x = "year", y = "budget", data=check)

plt.xticks(rotation = 90)

plt.show()

del check
budget["year_category"] = budget["year"].apply(lambda x: "Before 1955" if x<=1955 else ("1955 to 1990" if x<= 1990 else "Post 1990"))
avg_bud = pd.pivot_table(budget[budget["year"] != 1927], index="year_category", values="budget", aggfunc="mean").reset_index()
avg_bud
budget = pd.merge(budget, avg_bud, on="year_category", how="left")

budget["Modified_budget"] = budget[["budget_x", "budget_y"]].apply(lambda x: x[1] if x[0] == 0 else x[0], axis = 1) 
input_Data = pd.merge(input_Data, budget[["identifier", "Modified_budget"]], on="identifier", how="left")
input_Data = input_Data.drop("budget", axis = 1)

del budget

del avg_bud

del cluster

del score

del model
input_Data["language_extract"] = input_Data[["original_language","spoken_languages"]].apply(lambda x: [x[0]] if len(x[1]) == 0 else [i["iso_639_1"] for i in x[1]], axis = 1)

input_Data["language_coverage"] = input_Data["language_extract"].apply(lambda x: len(x))
sns.countplot("language_coverage", data = input_Data)

plt.show()
mlb = MultiLabelBinarizer()

input_Data = input_Data.join(pd.DataFrame(mlb.fit_transform(input_Data["language_extract"]), index = input_Data.index, columns = mlb.classes_))
input_Data = input_Data.drop(["language_extract","spoken_languages"], axis = 1)
input_Data["country_extract"] = input_Data["production_countries"].apply(lambda x: ["country na"] if len(x) == 0 else [i["name"] for i in x])

input_Data["country_coverage"] = input_Data["country_extract"].apply(lambda x: 0 if 'country na' in x else len(x))
sns.countplot("country_coverage", data=input_Data)

plt.show()
#mlb = MultiLabelBinarizer()

#language_check = input_Data[["year", "original_language"]].join(pd.DataFrame(mlb.fit_transform(input_Data["Country_extract"]), columns = mlb.classes_))

#language_check_inter = language_check.groupby(["year", "original_language"]).sum().reset_index()

#language_check_inter = language_check_inter.melt(id_vars=["year", "original_language"], var_name="Country")

#language_check_inter = language_check_inter[language_check_inter["value"] != 0]

#language_check_inter = language_check_inter.sort_values(["year","original_language","value"], ascending = False).drop_duplicates(["year", "original_language"])
#input_Data = pd.merge(input_Data, language_check_inter[["year", "original_language", "Country"]], on=["year", "original_language"], how="left")

#input_Data["Country_extract_mod"] = input_Data[["Country_extract", "Country"]].apply(lambda x:  [x[1]] if "Not Available" in x[0] else x[0], axis = 1)

#input_Data.at[1757,"Country_extract_mod"] = ["Vietnam"]

#input_Data.at[2342,"Country_extract_mod"] = ["Turkey"]

#input_Data.at[3939,"Country_extract_mod"] = ["Germany"]

#input_Data.at[5526,"Country_extract_mod"] = ["UnitedArabEmirates"]

#input_Data.at[6153,"Country_extract_mod"] = ["Turkey"]

#input_Data["Country_coverage"] = input_Data["Country_coverage"].apply(lambda x: 1 if x == 0 else x)
#sns.countplot("Country_coverage", data=input_Data)

#plt.show()
mlb = MultiLabelBinarizer()

input_Data = input_Data.join(pd.DataFrame(mlb.fit_transform(input_Data["country_extract"]), index = input_Data.index, columns = mlb.classes_))

input_Data = input_Data.drop(["country_extract","production_countries"],axis = 1)
np.isnan(input_Data["runtime"]).sum()
yearly_runtime = input_Data[np.isnan(input_Data["runtime"]) == False][["year","runtime"]]

yearly_runtime = pd.pivot_table(yearly_runtime, index="year", values="runtime", aggfunc="mean").reset_index()
plt.figure(figsize=(15,6))

sns.barplot(x="year",y = "runtime", data = yearly_runtime)

plt.xticks(rotation = 90)

plt.show()

del yearly_runtime
input_Data["runtime"] = input_Data["runtime"].fillna(np.mean(input_Data["runtime"]))
input_Data["prod_house_extract"] = input_Data["production_companies"].apply(lambda x: "prod house na" if len(x) == 0 else [i["name"] for i in x])

input_Data["prod_house_coverage"] = input_Data["prod_house_extract"].apply(lambda x: 0 if 'prod house na' in x else len(x))
mlb = MultiLabelBinarizer()

input_Data = input_Data.join(pd.DataFrame(mlb.fit_transform(input_Data["prod_house_extract"]),columns=mlb.classes_, index = input_Data.index))

input_Data = input_Data.drop(["prod_house_extract","production_companies"], axis = 1)
input_Data["crew_count"] = input_Data.crew.apply(len)
crew_check = pd.pivot_table(input_Data, index="year", values="crew_count", aggfunc="mean").reset_index()

plt.figure(figsize=(15,6))

sns.barplot("year", "crew_count",data = crew_check)

plt.xticks(rotation = 90)

plt.show()
input_Data["crew_count"] = input_Data["crew_count"].fillna(input_Data["crew_count"].mean())
def get_crew(x,crew):

    crew_ret = []

    if crew == "director":

        if pd.isnull(x) == True:

            crew_ret.append("director not known")

        else:

            for i,name in enumerate(re.split("}",x)):

                if len(re.findall(r"Director'.*,",str(name))) != 0:

                    crew_ret.append(re.sub(r"[',\s\"\]-]","",re.split(":",str(re.findall(r"Director'.*,",name)))[1]))

                else:

                    continue

        if len(crew_ret) == 0:

            crew_ret.append("No Director")

    else:

        if pd.isnull(x) == True:

            crew_ret.append("producer not known")

        else:

            for i,name in enumerate(re.split("}",x)):

                if len(re.findall(r"Producer'.*,",str(name))) != 0:

                    crew_ret.append(re.sub(r"[',\s\"\]-]","",re.split(":",str(re.findall(r"Producer'.*,",name)))[1]))

                else:

                    continue

        if len(crew_ret) == 0:

            crew_ret.append("No Producer")

    return crew_ret
#input_Data["Directors"] = input_Data["crew"].apply(get_crew,crew = "director")

#input_Data["Producers"] = input_Data["crew"].apply(get_crew,crew = "producer")
#mlb = MultiLabelBinarizer()

#input_Data = input_Data.join(pd.DataFrame(mlb.fit_transform(input_Data["Directors"]),columns=mlb.classes_+"_director", index = input_Data.index))

#input_Data = input_Data.join(pd.DataFrame(mlb.fit_transform(input_Data["Producers"]),columns=mlb.classes_+"_producer", index = input_Data.index))

#input_Data = input_Data.drop(["Directors","Producers","crew"],axis = 1)
input_Data = input_Data.drop("crew",axis = 1)
input_Data["cast_count"] = input_Data.cast.apply(len)
crew_check = pd.pivot_table(input_Data, index="year", values="cast_count", aggfunc="mean").reset_index()

plt.figure(figsize=(15,6))

sns.barplot("year", "cast_count",data = crew_check)

plt.xticks(rotation = 90)

plt.show()
input_Data["cast_count"] = input_Data["cast_count"].fillna(input_Data["cast_count"].mean())
input_Data = input_Data.drop("cast",axis = 1)