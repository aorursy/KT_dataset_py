import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tabulate import tabulate

import bq_helper

import os
#https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names

usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

query = """SELECT year, gender, name, sum(number) as count FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""

data = usa_names.query_to_pandas_safe(query)

data.to_csv("usa_names_data.csv")
data.sample(5)
print("Number of rows and columns in the data: ",data.shape,"\n")

print("Number and Range of years available: ",len(data["year"].unique()),"years between ",data["year"].min(), "to ",data["year"].max())



print("Total number of applicants in the dataset: ",sum(data["count"]))

print("% of male applicants : ","{0:.2f}".format(sum(data["count"][data["gender"]=="M"])/sum(data["count"])))

print("% of female applicants :","{0:.2f}".format(sum(data["count"][data["gender"]=="F"])/sum(data["count"])),"\n")



print("Total number of unique names in the data set: ",len(data["name"].unique()))

print("Total number of unique male names in the data set: ",len(data["name"][data["gender"]=="M"].unique()))

print("Total number of unique female names in the data set: ",len(data["name"][data["gender"]=="F"].unique()),"\n")



print("\n Most popular male names of all time")

print(tabulate(data[data["gender"]=="M"].groupby('name', as_index=False).agg({"count": "sum"}).sort_values("count",ascending=False).reset_index(drop=True).head(5),headers='keys', tablefmt='psql'))



print("\n Most popular female names of all time")

print(tabulate(data[data["gender"]=="F"].groupby('name', as_index=False).agg({"count": "sum"}).sort_values("count",ascending=False).reset_index(drop=True).head(5),headers='keys', tablefmt='psql'))
data=data[data["year"]>=1998]

data_agg=data.groupby(["year"],as_index=False).agg({"count": "sum"})

ax=data_agg.plot('year', 'count', kind='bar', figsize=(17,5), color='#86bf91', zorder=2, width=0.85)

ax.set_xlabel("Year", labelpad=20, size=12)

# Set y-axis label

ax.set_ylabel("# of Applicants", labelpad=20, size=12)

ax.legend_.remove()
def plot_yearly_count(character_name):

    data_agg=data[data["name"]==character_name].groupby(["year"],as_index=False).agg({"count": "sum"})

    if len(data_agg)==0:

        print("No data available")

    else:

        year_df=pd.DataFrame()

        year_df["year"]=data["year"].unique()

        data_agg["key"]=1

        data_agg=pd.merge(year_df,data_agg,on=["year"],how="left")

        data_agg=data_agg.sort_values("year",ascending=True)

        ax=data_agg.plot('year', 'count', kind='bar', figsize=(17,5), color='#86bf91', zorder=2, width=0.85)

        # Switch off ticks

        ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)

        # Set x-axis label

        ax.set_xlabel("Year", labelpad=20, size=12)

        # Set y-axis label

        ax.set_ylabel("# of Applicants", labelpad=20, size=12)

        # Set title

        ax.set_title("Popularity of the name "+str(character_name)+" in the past 20 years")

        ax.legend_.remove()
plot_yearly_count("Daenerys")
plot_yearly_count("Khaleesi")
plot_yearly_count("Arya")
plot_yearly_count("Sansa")
plot_yearly_count("Tyrion")
plot_yearly_count("Brienne")
plot_yearly_count("Lyanna")
plot_yearly_count("Meera")
plot_yearly_count("Jon")

plot_yearly_count("Catelyn")

plot_yearly_count("Jaime")