import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

%matplotlib inline

pd.set_option('display.max_rows', 15)

df = pd.read_csv("../input/Kaggle.csv")

df
print("Keskmine elektrifitseerimine:", df["Electrification rate or population"].mean())

print("Max elektrifitseerimine:", df["Electrification rate or population"].max())

print("Min elektrifitseerimine:", df["Electrification rate or population"].min())







(df[["Id", "Female Suicide Rate 100k people", "Expected years of schooling - Years"]]

 .sort_values("Expected years of schooling - Years", ascending=False))
(df[["Id", "Female Suicide Rate 100k people", "Gender Inequality Index 2014", "Homicide rate per 100k people 2008-2012"]]

 .sort_values("Female Suicide Rate 100k people", ascending=False))
(df[["Id", "Female Suicide Rate 100k people", "Public expenditure on education Percentange GDP",

     "Public health expenditure percentage of GDP 2013"]]

 .sort_values("Public expenditure on education Percentange GDP", ascending=False))
(df[["Id", "Female Suicide Rate 100k people", "MaleSuicide Rate 100k people",

     "Mobile phone subscriptions per 100 people 2014", "Internet users percentage of population 2014",

    "Physicians per 10k people"]]

 .sort_values("Female Suicide Rate 100k people", ascending=False))
(df.groupby(["Female Suicide Rate 100k people"])["Mobile phone subscriptions per 100 people 2014",

 "Internet users percentage of population 2014"].mean())

## (df.groupby(["Female Suicide Rate 100k people"]).aggregate({"Mobile phone subscriptions per 100 people 2014" : ["sum", "mean", "median"],

## "Internet users percentage of population 2014" : ["sum", "mean", "median"]}))
df.plot("Id", legend = False);

df.plot.hist("Female Suicide Rate 100k people", bins=11, grid=False, rwidth=0.95, legend = False);
df.plot.scatter("Female Suicide Rate 100k people", "Mobile phone subscriptions per 100 people 2014", alpha=0.2);
df.plot.scatter("Female Suicide Rate 100k people", "Internet users percentage of population 2014", alpha=0.2);
df.plot.scatter("Female Suicide Rate 100k people", "Physicians per 10k people", alpha=0.2);
## Töö salvestamine

df.to_csv("tulemus.csv")