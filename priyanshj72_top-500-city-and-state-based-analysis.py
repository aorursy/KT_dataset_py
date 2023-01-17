import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



cities = pd.read_csv("../input/cities_r2.csv")

cities.head(2)
cities["state_name"].value_counts().plot(kind="bar")
population_cities = cities.groupby("state_name")["population_total"].sum()

population_cities.sort(ascending=False)

population_cities.plot(kind="bar",color=['Red','Blue','Yellow'])
no_of_literates = cities.groupby("state_name")["literates_total"].sum()

no_of_literates.sort(ascending=False)

no_of_literates.plot(kind="bar",color=["Red","Blue","Yellow","Green"])
sex_ratio_states = cities.groupby("state_name")["sex_ratio"].mean()

sex_ratio_states.sort(ascending=False)

sex_ratio_states.plot(kind="bar",color=["blue","yellow","pink","brown","red"])
literacy_rates = cities.groupby("state_name")["effective_literacy_rate_total"].mean()

literacy_rates.sort(ascending=False)

literacy_rates.plot(kind="bar")
city_literates = cities.groupby("name_of_city")["effective_literacy_rate_total"].mean()

city_literates.sort(ascending=False)

top_twenty = city_literates[:20]

top_twenty.plot(kind="bar")
cities["graduate_percent"] = (cities["total_graduates"]/cities["population_total"])*100

cities.head()
city_graduate = cities.groupby("name_of_city")["graduate_percent"].sum()

city_graduate.sort(ascending=False)

top_twenty = city_graduate[:20]

top_twenty.plot(kind="bar")
city_sex_ratio = cities.groupby("name_of_city")["sex_ratio"].mean()

city_sex_ratio.sort(ascending=False)

top_twenty = city_sex_ratio[:20]

top_twenty.plot(kind="bar",color=["red","blue","yellow"])
city_child_sex_ratio = cities.groupby("name_of_city")["child_sex_ratio"].mean()

city_child_sex_ratio.sort(ascending=False)

top_twenty = city_child_sex_ratio[:20]

top_twenty.plot(kind="bar",color=["red","blue","yellow"])