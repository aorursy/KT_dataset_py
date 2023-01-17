%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
#path for the datset file 

path_data = "../input/cities_r2.csv"

# reading csv file 

data = pd.read_csv(path_data)

data.head(5)
#extracting top 25 cities in terms of population

top_25_populated_city =  data.sort('population_total', ascending=False).head(25)[["name_of_city","population_total",

                                                                                 "population_male","population_female"]]

top_25_populated_city.head(5)
#plot a bar chart for the city name and population (y lim - 1.3 * 10^7) 

top_25_populated_city.plot(x = top_25_populated_city["name_of_city"],

                           kind='bar', figsize=[19, 8], width=0.5,

                           alpha=0.5, #color='r', edgecolor='k',

                           stacked = True,

                           grid=False)
top_25_0_6_age_populated_city =  data.sort('population_total', ascending=False).head(25)[["name_of_city",

                                                                                  "0-6_population_total",

                                                                                 "0-6_population_male",

                                                                                  "0-6_population_female"]]

top_25_0_6_age_populated_city.head(5)
top_25_0_6_age_populated_city.plot(x = top_25_populated_city["name_of_city"],

                           kind='bar', figsize=[19, 8], width=0.5,

                           alpha=0.5, #color='r', edgecolor='k',

                           stacked = True,

                           grid=False)
state_sex_ratio = data[["state_name","sex_ratio"]].groupby("state_name").agg({"sex_ratio":np.average}).sort("sex_ratio")

state_sex_ratio.plot(kind="barh",

                      grid=False,

                      figsize=(16,10),

                      color="b",

                      alpha = 0.5,

                      width=0.6,

                     edgecolor="g",)
state_sex_ratio = data[["state_name","child_sex_ratio"]].groupby("state_name").agg({"child_sex_ratio":np.average}).sort("child_sex_ratio")

state_sex_ratio.plot(kind="barh",

                      grid=False,

                      figsize=(16,10),

                      color="g",

                      alpha = 0.5,

                      width=0.6,

                     edgecolor="g",)
state_literacy_effective  = data[["state_name",

                                  "effective_literacy_rate_total",

                                  "effective_literacy_rate_male",

                                  "effective_literacy_rate_female"]].groupby("state_name").agg({"effective_literacy_rate_total":np.average,

                                                                                                "effective_literacy_rate_male":np.average,

                                                                                                "effective_literacy_rate_female":np.average}).sort("effective_literacy_rate_total", ascending=False)

state_literacy_effective.plot(kind="bar",

                      grid=False,

                      figsize=(16,10),

                      #color="r",

                      alpha = 0.5,

                      width=0.6,

                      stacked = False,

                     edgecolor="g",)
state_graduate  = data[["state_name",

                                  "total_graduates",

                                  "male_graduates",

                                  "female_graduates"]].groupby("state_name").agg({"total_graduates":np.sum,

                                                                                "male_graduates":np.sum,

                                                                                "female_graduates":np.sum}).sort("total_graduates", ascending=False)

state_graduate.plot(kind="bar",

                      grid=False,

                      figsize=(16,10),

                      #color="r",

                      alpha = 0.5,

                      width=0.6,

                      stacked = False,

                     edgecolor="g",)