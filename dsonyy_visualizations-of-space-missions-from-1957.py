import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sb
missions = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv", index_col=0)
missions.head()
missions.info()
missions["Datum"] = pd.to_datetime(missions["Datum"])

missions["Year"] = [date.year for date in missions["Datum"]]

missions["Rocket"] = missions[" Rocket"].str.replace(",", "").astype("float32")

missions = missions.drop(columns=["Unnamed: 0.1", " Rocket"])

missions
missions.info()
def successed_and_failed(missions):

    year_begin =  min(missions["Year"])

    year_end = max(missions["Year"])



    missions_years = missions["Year"].value_counts().sort_index()

    failure_missions_years = missions.loc[missions["Status Mission"] == "Failure"]["Year"].value_counts().sort_index()



    plt.subplots(figsize=(18,12))

    plt.bar(missions_years.index, missions_years.values, 

        edgecolor="k", width=1, label="Success")

    plt.bar(failure_missions_years.index, failure_missions_years.values,

        edgecolor="k", width=1, label="Failure / Partial Failure")

    plt.xticks(range(year_begin, year_end + 1), rotation=90)

    plt.ylim((0, 130))

    plt.xlabel("Year")

    plt.ylabel("Missions")

    plt.legend()

    plt.title("Successes and failures of space missions (Oct 1957 - Aug 2020)",

        fontdict={"fontsize":20})



    for year in range(year_begin, year_end + 1):

        plt.annotate(f"{missions_years[year] - failure_missions_years[year]}",

            (year - 0.25, missions_years[year] + 1), rotation=90)

        plt.annotate(f"{failure_missions_years[year]}",

            (year - 0.25, failure_missions_years[year] + 1), rotation=90)



successed_and_failed(missions)
def faiure_rate(missions):

    year_begin = min(missions["Year"])

    year_end = max(missions["Year"])



    missions_years = missions["Year"].value_counts().sort_index()

    failure_missions_years = missions.loc[missions["Status Mission"] != "Success"]["Year"].value_counts().sort_index()

    failure_rate = failure_missions_years / missions_years



    plt.subplots(figsize=(18,12))

    plt.bar(failure_rate.index, failure_rate.values, edgecolor="k", width=1, color="r")

    plt.xticks(range(year_begin, year_end + 1), rotation=90)

    plt.gca().set_yticklabels(["{:,.0%}".format(x) for x in np.arange(0, 1, 0.1)])

    plt.xlabel("Year")

    plt.ylabel("Failure percent")

    plt.title("Failure and partial failure rate of space missions (Oct 1957 - Aug 2020)", fontdict={"fontsize":20})

    plt.grid(False)



    for year in range(year_begin, year_end + 1):

        plt.annotate(f"{failure_rate[year]*100:.0f}", 

            (year - 0.25, failure_rate[year] + 0.01), rotation=90)



faiure_rate(missions)
def costs(missions):

    year_begin =  min(missions["Year"])

    year_end = max(missions["Year"])



    missions_cost = missions.groupby(["Year"])["Rocket"].sum().sort_index() / 1000



    plt.subplots(figsize=(18,12))

    plt.bar(missions_cost.index, missions_cost.values, 

        edgecolor="k", width=1, label="Success", color="orange")

    plt.xticks(range(year_begin, year_end + 1), rotation=90)

    plt.xlabel("Year")

    plt.ylabel("Total Const in Billions of Dollars")

    plt.ylim((0, 6.5))

    plt.title("Total cost of space missions in billions of dollars (Oct 1957 - Aug 2020)\n"

        "INCOMPLETE DATA", fontdict={"fontsize":20})



    for year in range(year_begin, year_end + 1):

        plt.annotate(f"{missions_cost[year]:.1f}", 

            (year - 0.25, missions_cost[year] + 0.1), rotation=90)



costs(missions)
def active_companies(missions):

    year_begin =  min(missions["Year"])

    year_end = max(missions["Year"])



    active_companies = missions.groupby(["Year"])["Company Name"].nunique()



    plt.subplots(figsize=(18,12))

    plt.bar(active_companies.index, active_companies.values, 

        edgecolor="k", width=1, label="Success", color="lightgreen")

    plt.xticks(range(year_begin, year_end + 1), rotation=90)

    plt.yticks(range(0, 25))

    plt.xlabel("Year")

    plt.ylabel("Number of Companies")

    plt.title("Number of companies which launched at least one space misssion (Oct 1957 - Aug 2020)",

        fontdict={"fontsize":20})

    for year in range(year_begin, year_end + 1):

        plt.annotate(f"{active_companies[year]}", 

            (year - 0.25, active_companies[year] + 0.3), rotation=90)



active_companies(missions)
def top_companies(missions, since, to, n=12):

    top_companies = missions.loc[(missions["Year"] < to) & (missions["Year"] >= since)]["Company Name"].value_counts()[:n].sort_values()



    plt.subplots(figsize=(16,12))

    plt.barh(top_companies.index, top_companies.values, edgecolor='k', color="blue")

    plt.title(f"Top {n} companies by the number of space missions ({since}-{to})", {"fontsize":20})

    plt.xlabel("Number of missions")

    plt.ylabel("Company Name")

    plt.xlim((0, 1300))

    for i in range(n):

        plt.annotate(f"  {top_companies.iloc[i]}", (top_companies[i], i - 0.1))



top_companies(missions, 1957, 1979)
top_companies(missions, 1980, 1999)
top_companies(missions, 2000, 2020)
def top_companies_costs(missions, since, to, n=12):

    years = (missions["Year"] < to) & (missions["Year"] >= since)

    top_companies_costs = missions[years].groupby(missions["Company Name"])["Rocket"].sum().sort_values()[-n:]



    plt.subplots(figsize=(16,12))

    plt.barh(top_companies_costs.index, top_companies_costs.values, edgecolor='k', color="orange")

    plt.title(f"Top {n} companies by their costs ({since}-{to - 1})\nINCOMPLETE DATA", {"fontsize":20})

    plt.xlabel("Total Cost")

    plt.ylabel("Company Name")

    plt.xlim((0, 46000))

    for i in range(n):

        plt.annotate(f"  {top_companies_costs.iloc[i]:.1f}", (top_companies_costs[i], i - 0.1))



        

top_companies_costs(missions, 1957, 1989)
top_companies_costs(missions, 1980, 2000)
top_companies_costs(missions, 2000, 2021)