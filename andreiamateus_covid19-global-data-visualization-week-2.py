# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline



#import libraries for Choropleth

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import seaborn as sb

from datetime import datetime, timedelta



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

forecast = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

train
# What's the inverval of the data?

print(train["Date"].min())

print(train["Date"].max())



# Create variable for last day with format Day-Month-Year:

first_date = train["Date"].min()

last_date = train["Date"].max()

last_date_f = datetime.strptime(train["Date"].max(), "%Y-%m-%d").strftime("%d-%B-%Y")

second2last_date = (datetime.strptime(train["Date"].max(), "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
# How many countries are in the data and which ones are they?

print(len(train["Country_Region"].unique()))

countries = train["Country_Region"].unique()

countries
# Last worldwide results into variables:

worldwide_evolution_i = train.drop(["Id"], axis=1).groupby("Date").sum()



nbr_ww_confirmed_cases = worldwide_evolution_i.loc[train["Date"].max()][0]

new_daily_ww_confirmed_cases = worldwide_evolution_i.loc[last_date][0] - worldwide_evolution_i.loc[second2last_date][0]

growth_factor_ww_confirmed_cases = (((worldwide_evolution_i.loc[last_date][0] / worldwide_evolution_i.loc[second2last_date][0])-1)*100).round(2)

nbr_ww_fatalities = worldwide_evolution_i.loc[train["Date"].max()][1]

new_daily_ww_fatalities = worldwide_evolution_i.loc[last_date][1] - worldwide_evolution_i.loc[second2last_date][1] 

growth_factor_ww_fatalities = (((worldwide_evolution_i.loc[last_date][1] / worldwide_evolution_i.loc[second2last_date][1])-1)*100).round(2)

    

cfr_ww = (nbr_ww_fatalities/nbr_ww_confirmed_cases)*100
# Evolution table per Country & per Date:

perCountry_evolution_i = train.drop(["Id"], axis=1).groupby(["Country_Region", "Date"]).sum()
def get_last_WW_numbers():

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("   COVID-19 Coronavirus Pandemic - Worldwide Numbers", fontsize=18, fontweight='bold', COLOR="DARKBLUE")

    sb.set(style="white")



    ax = fig.add_subplot(111)

    ax.set_title("Update: {}".format(last_date_f), ha="center")

    plt.xticks([0,10], " ")

    plt.yticks([0,11], " ")

    ax.spines['bottom'].set_color('grey')

    ax.spines['top'].set_color('grey') 

    ax.spines['right'].set_color('white')

    ax.spines['left'].set_color('white')



    ax.tick_params(right= False,top= False,left= False, bottom= False)

    ax.text(5, 10.3, "Number of Confirmed Cases:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 9.8, "{:,.0f}".format(nbr_ww_confirmed_cases), fontsize=20, fontweight='bold', COLOR="darkorange", ha="center")

    ax.text(5, 8.8, "New Daily Confirmed Cases:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 8.3, "+{:,.0f} ".format(new_daily_ww_confirmed_cases), fontsize=20, fontweight='bold', COLOR="darkorange", ha="center")

    ax.text(5, 7.3, "Growth Factor of Confirmed Cases:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 6.8, "{:.2f}%".format(growth_factor_ww_confirmed_cases), fontsize=20, fontweight='bold', COLOR="darkgray", ha="center")

    ax.text(5, 5.3, "Number of Fatalities:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 4.8, "{:,.0f}".format(nbr_ww_fatalities), fontsize=20, fontweight='bold', COLOR="darkred", ha="center")

    ax.text(5, 3.8, "New Daily Fatalities:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 3.3, "+{:,.0f} ".format(new_daily_ww_fatalities), fontsize=20, fontweight='bold', COLOR="darkred", ha="center")

    ax.text(5, 2.3, "Growth Factor of Fatalities:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 1.8, "{:.2f}%".format(growth_factor_ww_fatalities), fontsize=20, fontweight='bold', COLOR="darkgray", ha="center")

    ax.text(5, 0.8, "Worldwide Case Fatality Rate (%):", fontsize=16, fontweight='bold', COLOR="black", ha="center")

    ax.text(5, 0.3, "{:.1f}".format(cfr_ww), fontsize=20, fontweight='bold', COLOR="darkblue", ha="center")

    



def get_last_numbers_perCountry(country):

    

    if country in train["Country_Region"].unique():

    

        nbr_confirmed_cases_perCountry = perCountry_evolution_i.loc[country].loc[last_date][0]

        new_daily_confirmed_cases_perCountry = perCountry_evolution_i.loc[country].loc[last_date][0] - perCountry_evolution_i.loc[country].loc[second2last_date][0]

        growth_factor_confirmed_cases_perCountry = (((perCountry_evolution_i.loc[country].loc[last_date][0] / perCountry_evolution_i.loc[country].loc[second2last_date][0])-1)*100).round(2)

        nbr_fatalities_perCountry = perCountry_evolution_i.loc[country].loc[last_date][1]

        new_daily_fatalities_perCountry = perCountry_evolution_i.loc[country].loc[last_date][1] - perCountry_evolution_i.loc[country].loc[second2last_date][1] 

        growth_factor_fatalities_perCountry = (((perCountry_evolution_i.loc[country].loc[last_date][1] / perCountry_evolution_i.loc[country].loc[second2last_date][1])-1)*100).round(2)



        cfr_perCountry = (nbr_fatalities_perCountry/nbr_confirmed_cases_perCountry)*100



        fig = plt.figure(figsize=(10,10))

        fig.suptitle("   COVID-19 Coronavirus Pandemic - {} Numbers".format(country), fontsize=18, fontweight='bold', COLOR="DARKBLUE")

        sb.set(style="white")



        ax = fig.add_subplot(111)

        ax.set_title("Update: {}".format(last_date_f), ha="center")

        plt.xticks([0,10], " ")

        plt.yticks([0,11], " ")

        ax.spines['bottom'].set_color('grey')

        ax.spines['top'].set_color('grey') 

        ax.spines['right'].set_color('white')

        ax.spines['left'].set_color('white')



        ax.tick_params(right= False,top= False,left= False, bottom= False)

        ax.text(5, 10.3, "Number of Confirmed Cases:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 9.8, "{:,.0f}".format(nbr_confirmed_cases_perCountry), fontsize=20, fontweight='bold', COLOR="darkorange", ha="center")

        ax.text(5, 8.8, "New Daily Confirmed Cases:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 8.3, "+{:,.0f} ".format(new_daily_confirmed_cases_perCountry), fontsize=20, fontweight='bold', COLOR="darkorange", ha="center")

        ax.text(5, 7.3, "Growth Factor of Confirmed Cases:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 6.8, "{:.2f}%".format(growth_factor_confirmed_cases_perCountry), fontsize=20, fontweight='bold', COLOR="darkgray", ha="center")

        ax.text(5, 5.3, "Number of Fatalities:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 4.8, "{:,.0f}".format(nbr_fatalities_perCountry), fontsize=20, fontweight='bold', COLOR="darkred", ha="center")

        ax.text(5, 3.8, "New Daily Fatalities:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 3.3, "+{:,.0f} ".format(new_daily_fatalities_perCountry), fontsize=20, fontweight='bold', COLOR="darkred", ha="center")

        ax.text(5, 2.3, "Growth Factor of Fatalities:", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 1.8, "{:.2f}%".format(growth_factor_fatalities_perCountry), fontsize=20, fontweight='bold', COLOR="darkgray", ha="center")

        ax.text(5, 0.8, "Country Case Fatality Rate (%):", fontsize=16, fontweight='bold', COLOR="black", ha="center")

        ax.text(5, 0.3, "{:.1f}".format(cfr_perCountry), fontsize=20, fontweight='bold', COLOR="darkblue", ha="center")    

    

    else:

        return "Country does not exist"
# Worldwide evolution with addition of new metrics:

worldwide_evolution = worldwide_evolution_i.reset_index()



CFR = pd.DataFrame({"CFR (%)": (worldwide_evolution["Fatalities"] / worldwide_evolution["ConfirmedCases"] * 100)})

CFR["CFR (%)"] = CFR["CFR (%)"].fillna(0)



days = worldwide_evolution_i.index



ww_new_daily_cases = []

ww_new_daily_fatalities = []

ww_growth_factor_cases = []

ww_growth_factor_fatalities = []



for i in days:

    yesterday = (datetime.strptime(i, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    

    if yesterday in days:

        delta_cc = worldwide_evolution_i.loc[i]["ConfirmedCases"]-worldwide_evolution_i.loc[yesterday]["ConfirmedCases"]

        delta_fa = worldwide_evolution_i.loc[i]["Fatalities"]-worldwide_evolution_i.loc[yesterday]["Fatalities"]

        growth_cc = ((worldwide_evolution_i.loc[i]["ConfirmedCases"]/worldwide_evolution_i.loc[yesterday]["ConfirmedCases"])-1)*100

        growth_fa = ((worldwide_evolution_i.loc[i]["Fatalities"]/worldwide_evolution_i.loc[yesterday]["Fatalities"])-1)*100

        

        ww_new_daily_cases.append(delta_cc)

        ww_new_daily_fatalities.append(delta_fa)

        ww_growth_factor_cases.append(growth_cc)

        ww_growth_factor_fatalities.append(growth_fa)



    else:

        delta = 0

        ww_new_daily_cases.append(delta)

        ww_new_daily_fatalities.append(delta)

        ww_growth_factor_cases.append(delta)

        ww_growth_factor_fatalities.append(delta)





new_cases = pd.DataFrame({"New Confirmed Cases": ww_new_daily_cases, "New Fatalities" : ww_new_daily_fatalities})

growth_evolution = pd.DataFrame({"Growth Factor Confirmed Cases (%)" : ww_growth_factor_cases, "Growth Factor Fatalities (%)" : ww_growth_factor_fatalities})



worldwide_evolution_wGF = worldwide_evolution.join(CFR.round(2), how="right").join(new_cases,how="right").join(growth_evolution.round(2),how="right")
worldwide_evolution_wGF
# Per Country evolution with addition of new metrics:

perCountry_evolution = perCountry_evolution_i.reset_index()



CFR_perCountry = pd.DataFrame({"CFR (%)": (perCountry_evolution["Fatalities"] / perCountry_evolution["ConfirmedCases"] * 100)})

CFR_perCountry["CFR (%)"] = CFR_perCountry["CFR (%)"].fillna(0)



pc_new_daily_cases = []

pc_new_daily_fatalities = []

pc_growth_factor_cases = []

pc_growth_factor_fatalities = []



for c in countries:

    for i in days:

        yesterday = (datetime.strptime(i, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")



        if yesterday in days:

            delta_cc = perCountry_evolution_i.loc[c].loc[i]["ConfirmedCases"]-perCountry_evolution_i.loc[c].loc[yesterday]["ConfirmedCases"]

            delta_fa = perCountry_evolution_i.loc[c].loc[i]["Fatalities"]-perCountry_evolution_i.loc[c].loc[yesterday]["Fatalities"]

            growth_cc = ((perCountry_evolution_i.loc[c].loc[i]["ConfirmedCases"]/perCountry_evolution_i.loc[c].loc[yesterday]["ConfirmedCases"])-1)*100

            growth_fa = ((perCountry_evolution_i.loc[c].loc[i]["Fatalities"]/perCountry_evolution_i.loc[c].loc[yesterday]["Fatalities"])-1)*100



            pc_new_daily_cases.append(delta_cc)

            pc_new_daily_fatalities.append(delta_fa)

            pc_growth_factor_cases.append(growth_cc)

            pc_growth_factor_fatalities.append(growth_fa)



        else:

            delta = 0

            pc_new_daily_cases.append(delta)

            pc_new_daily_fatalities.append(delta)

            pc_growth_factor_cases.append(delta)

            pc_growth_factor_fatalities.append(delta)





pc_new_cases = pd.DataFrame({"New Confirmed Cases": pc_new_daily_cases, "New Fatalities" : pc_new_daily_fatalities})

pc_growth_evolution = pd.DataFrame({"Growth Factor Confirmed Cases (%)" : pc_growth_factor_cases, "Growth Factor Fatalities (%)" : pc_growth_factor_fatalities})

pc_growth_evolution["Growth Factor Confirmed Cases (%)"] = pc_growth_evolution["Growth Factor Confirmed Cases (%)"].fillna(0)

pc_growth_evolution["Growth Factor Fatalities (%)"] = pc_growth_evolution["Growth Factor Fatalities (%)"].fillna(0)

perCountry_evolution_wGF = perCountry_evolution.join(CFR_perCountry.round(2), how="right").join(pc_new_cases,how="right").join(pc_growth_evolution.round(2),how="right")
perCountry_evolution_wGF
# Per Country last numbers with addition of new metrics:

perCountry_evolution_wGF_byDate = perCountry_evolution_wGF.set_index(["Date","Country_Region"])

last_day = perCountry_evolution_wGF_byDate.loc[train["Date"].max()]



# Group all provinces/states of each country

last_day_perCountry = last_day.groupby("Country_Region").sum()

last_day_perCountry_ri = last_day_perCountry.reset_index()

last_day_perCountry_ri
################################################# Plot worldwide evolution ##############################################################

def plot_WW_evolution():

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="ConfirmedCases", data=worldwide_evolution, linewidth=2, color='darkblue', marker='o', markersize=6)

    ax = sb.lineplot(x="Date", y="Fatalities", data=worldwide_evolution, linewidth=2, color='darkred', marker='o', markersize=6)

    plt.title("COVID19 - Worlwide Total Numbers", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=["Confirmed Cases", "Fatalities"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()

    

def plot_ww_CFR_rate():

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="CFR (%)", data=worldwide_evolution_wGF, linewidth=2, color='darkblue', marker='o', markersize=6)

    plt.title("COVID19 - Worlwide Case Fatality Rate (%)", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("CFR (%)", fontsize=12)

    plt.legend(labels=["CFR (%)"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.show()

    

def plot_ww_newCases():

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="New Confirmed Cases", data=worldwide_evolution_wGF, linewidth=2, color='darkblue', marker='o', markersize=6)

    ax = sb.lineplot(x="Date", y="New Fatalities", data=worldwide_evolution_wGF, linewidth=2, color='darkred', marker='o', markersize=6)

    plt.title("COVID19 - Worlwide New Daily Cases", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=["New Confirmed Cases", "New Fatalities"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()

    

def plot_ww_growthFactor():

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="Growth Factor Confirmed Cases (%)", data=worldwide_evolution_wGF, linewidth=2, color='darkblue', marker='o', markersize=6)

    ax = sb.lineplot(x="Date", y="Growth Factor Fatalities (%)", data=worldwide_evolution_wGF, linewidth=2, color='darkred', marker='o', markersize=6)

    plt.title("COVID19 - Worlwide Growth Factor", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("%", fontsize=12)

    plt.legend(labels=["New Confirmed Cases", "New Fatalities"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    ax.set_ylim([None,50])

    plt.show()

    

# Plot last Confirmed cases & fatalites per country ordered by worst cases:    

def plot_confirmedCases_top30countries():

    confirmed_cases = last_day_perCountry_ri.sort_values("ConfirmedCases", ascending = False)

    plt.figure(figsize=(14,10))

    sb.set(style="darkgrid")

    ax = sb.barplot(y="Country_Region", x="ConfirmedCases", data=confirmed_cases.head(30))



    plt.title("COVID19 TOP30 Countries - Total Confirmed Cases [" + last_date_f + "]", fontsize = 16)

    plt.xlabel("Counts", fontsize=12)

    plt.ylabel("Country", fontsize=12)

    plt.xticks(rotation=60)



    for p in ax.patches:

        ax.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()+1.2),

                xytext=(5, 10), textcoords='offset points')



def plot_fatalities_top30countries():

    fatalities = last_day_perCountry_ri.sort_values("Fatalities", ascending = False)

    plt.figure(figsize=(14,10))

    sb.set(style="darkgrid")

    ax = sb.barplot(y="Country_Region", x="Fatalities", data=fatalities.head(30))



    last_date_f = datetime.strptime(train["Date"].max(), "%Y-%m-%d").strftime("%d-%B-%Y")

    plt.title("COVID19 TOP30 Countries - Total Fatalities [" + last_date_f + "]", fontsize = 16)

    plt.xlabel("Counts", fontsize=12)

    plt.ylabel("Country", fontsize=12)

    plt.xticks(rotation=60)



    for p in ax.patches:

        ax.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()+1.2),

                xytext=(5, 10), textcoords='offset points')

        

# Plot last CFR rate for countries with higher confirmed cases (>500):

def plot_last_cfr_forTopCountries():

    CFR_sorted_perCountry = last_day_perCountry_ri.sort_values("CFR (%)", ascending = False)

    CFR_sorted_forTopCountries = CFR_sorted_perCountry[CFR_sorted_perCountry["ConfirmedCases"] >1000]

    CFR_sorted_forTopCountries



    plt.figure(figsize=(14,10))

    sb.set(style="darkgrid")

    ax = sb.barplot(y="Country_Region", x="CFR (%)", data=CFR_sorted_forTopCountries.head(30))



    plt.title("COVID19 - Worst Case Fatality Rate per Country [" + last_date_f + "]", fontsize = 16)

    plt.xlabel("CFR (%)", fontsize=12)

    plt.ylabel("Country", fontsize=12)

    plt.xticks(rotation=60)

    for p in ax.patches:

          ax.annotate("%.2f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()+1.2),

                    xytext=(5, 10), textcoords='offset points')

    ax.text(6.2, 25, "* only countries with at least 1000 \n   confirmed cases were taken into \n    account", fontsize=14)





############################################# Plot Per Country evolution #################################################################

perCountry_evolution_woDate = perCountry_evolution_wGF.set_index(["Country_Region"])

    

def plot_perCountry_evolution(country): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="ConfirmedCases", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkblue', marker='o', markersize=6)

    ax = sb.lineplot(x="Date", y="Fatalities", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkred', marker='o', markersize=6)

    plt.title("COVID19 - {} Total Numbers".format(country), fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=["Confirmed Cases", "Fatalities"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()

    

def plot_perCountry_CFR_rate(country):

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="CFR (%)", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkblue', marker='o', markersize=6)

    plt.title("COVID19 - {} Case Fatality Rate (%)".format(country), fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("CFR (%)", fontsize=12)

    plt.legend(labels=["CFR (%)"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.show()

    

def plot_perCountry_newCases(country):

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="New Confirmed Cases", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkblue', marker='o', markersize=6)

    ax = sb.lineplot(x="Date", y="New Fatalities", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkred', marker='o', markersize=6)

    plt.title("COVID19 - {} New Daily Cases".format(country), fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=["New Confirmed Cases", "New Fatalities"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()



def plot_perCountry_growthFactor(country):

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    ax = sb.lineplot(x="Date", y="Growth Factor Confirmed Cases (%)", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkblue', marker='o', markersize=6)

    ax = sb.lineplot(x="Date", y="Growth Factor Fatalities (%)", data=perCountry_evolution_woDate.loc[country], linewidth=2, color='darkred', marker='o', markersize=6)

    plt.title("COVID19 - {} Growth Factor".format(country), fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("%", fontsize=12)

    plt.legend(labels=["New Confirmed Cases", "New Fatalities"], fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    #ax.set_ylim([None,50])

    plt.show()

############################################# Plot Comparitive charts for different countries #################################################################

def plot_comparative_ConfirmedCases(countryList):    

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="ConfirmedCases", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - Confirmed Cases Evolution", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()

    

def plot_comparative_Fatalities(countryList): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="Fatalities", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - Fatalities Evolution", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()

    

def plot_comparative_CFR(countryList): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="CFR (%)", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))    



    plt.title("COVID19 - Case Fatality Rate Evolution (%)", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("(%)", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



    plt.show()

    

def plot_comparative_NewConfirmedCases(countryList): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="New Confirmed Cases", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))    

        

    plt.title("COVID19 - New Confirmed Cases", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("Count", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()

    

def plot_comparative_NewFatalities(countryList): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="New Fatalities", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))    

            

    plt.title("COVID19 - New Fatalities", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("Count", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()

    

def plot_comparative_GF_ConfirmedCases(countryList): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="Growth Factor Confirmed Cases (%)", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))    

            

    plt.title("COVID19 - Growth Factor of Confirmed Cases (%)", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("(%)", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



    plt.show()

    

def plot_comparative_GF_Fatalities(countryList): 

    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')



    for c in countryList:

        if c in train["Country_Region"].unique():

            ax = sb.lineplot(x="Date", y="Growth Factor Fatalities (%)", data=perCountry_evolution_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))   

            

    plt.title("COVID19 - Growth Factor of Fatalities (%)", fontsize = 16)

    plt.xlabel("Date", fontsize=12)

    plt.ylabel("(%)", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



    plt.show()





############################################# Plot Comparitive charts for different countries #################################################################

def plot_comparative_CC_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="ConfirmedCases", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - Confirmed Cases Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()



def plot_comparative_F_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="Fatalities", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - Fatalities Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()



def plot_comparative_CFR_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="CFR (%)", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - CFR (%) Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("(%)", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



    plt.show()

    

def plot_comparative_NewCC_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="New Confirmed Cases", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - New Confirmed Cases Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()

    

def plot_comparative_GrowthCC_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="Growth Factor Confirmed Cases (%)", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - Growth Factor Confirmed Cases (%) Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("(%)", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



    plt.show()



def plot_comparative_NewF_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="New Fatalities", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - New Fatalities Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("Counts", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))



    plt.show()

    

def plot_comparative_GrowthF_sincePatient(patient,countryList): 

    perCountry_evolution_patient=perCountry_evolution_wGF[perCountry_evolution_wGF["ConfirmedCases"]>=patient]

    perCountry_evolution_patient_woDate = perCountry_evolution_patient.set_index(["Country_Region"])



    plt.figure(figsize=(14,8))

    sb.set(style="darkgrid")

    palette = plt.get_cmap('Dark2')

    

    for c in countryList:

        if c in train["Country_Region"].unique():

            x= (len(perCountry_evolution_patient_woDate.loc[c])+1)

            ax = sb.lineplot(x=np.arange(1,x,1), y="Growth Factor Fatalities (%)", data=perCountry_evolution_patient_woDate.loc[c], linewidth=2, marker='o', markersize=6)

        else: 

            print("{} coutry doesn't exist!".format(c))

            

    plt.title("COVID19 - Growth Factor Fatalities (%) Evolution", fontsize = 16)

    plt.xlabel("Days since patient #{}".format(patient), fontsize=12)

    #ax.set_yscale('log')

    plt.ylabel("(%)", fontsize=12)

    plt.legend(labels=countryList, fontsize = 12)

    plt.xticks(rotation=60)

    ax=plt.axes()                                                                                                                                                                    

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



    plt.show()
# Create the Choropleth

def confirmedCases_map():

    fig = go.Figure(data=go.Choropleth(locations = last_day_perCountry_ri['Country_Region'], locationmode = 'country names', z = last_day_perCountry_ri['ConfirmedCases'],

        colorscale = 'Reds', marker_line_color = 'black',marker_line_width = 0.5, 

    ))



    fig.update_layout(

        title_text = "Confirmed Cases as {}".format(last_date_f),

        title_x = 0.5,

        width=1400, 

        height=700,

        geo=dict(

            showframe = True,

            showcoastlines = False,

            projection_type = 'equirectangular'

        )

    )

    fig.show()

    

def new_confirmedCases_map():

    fig = go.Figure(data=go.Choropleth(locations = last_day_perCountry_ri['Country_Region'], locationmode = 'country names', z = last_day_perCountry_ri['New Confirmed Cases'],

        colorscale = 'Reds', marker_line_color = 'black',marker_line_width = 0.5, 

    ))



    fig.update_layout(

        title_text = "New Confirmed Cases as {}".format(last_date_f),

        title_x = 0.5,

        width=1400, 

        height=700,

        geo=dict(

            showframe = True,

            showcoastlines = False,

            projection_type = 'equirectangular'

        )

    )

    fig.show()

    

def fatalities_map():

    fig = go.Figure(data=go.Choropleth(locations = last_day_perCountry_ri['Country_Region'], locationmode = 'country names', z = last_day_perCountry_ri['Fatalities'],

        colorscale = 'Reds', marker_line_color = 'black',marker_line_width = 0.5, 

    ))



    fig.update_layout(

        title_text = "Fatalities as {}".format(last_date_f),

        title_x = 0.5,

        width=1400, 

        height=700,

        geo=dict(

            showframe = True,

            showcoastlines = False,

            projection_type = 'equirectangular'

        )

    )

    fig.show()

    

def new_fatalities_map():

    fig = go.Figure(data=go.Choropleth(locations = last_day_perCountry_ri['Country_Region'], locationmode = 'country names', z = last_day_perCountry_ri['New Fatalities'],

        colorscale = 'Reds', marker_line_color = 'black',marker_line_width = 0.5, 

    ))



    fig.update_layout(

        title_text = "New Fatalities as {}".format(last_date_f),

        title_x = 0.5,

        width=1400, 

        height=700,

        geo=dict(

            showframe = True,

            showcoastlines = False,

            projection_type = 'equirectangular'

        )

    )

    fig.show()
# Get last Worlwide results:

def worlwide_evolution():

    get_last_WW_numbers()

    plot_WW_evolution()

    plot_ww_newCases()

    plot_ww_growthFactor()

    plot_ww_CFR_rate()

    plot_confirmedCases_top30countries()

    plot_fatalities_top30countries()

    plot_last_cfr_forTopCountries()



# Get last perCountry results:

def perCountry_evolution(country):

    get_last_numbers_perCountry(country)

    plot_perCountry_evolution(country)

    plot_perCountry_newCases(country)

    plot_perCountry_growthFactor(country)

    plot_perCountry_CFR_rate(country)



# Plot comparative charts with countries input from user:

def comparative_plots(countryList):

    plot_comparative_ConfirmedCases(countryList)

    plot_comparative_Fatalities(countryList)

    plot_comparative_CFR(countryList)

    plot_comparative_NewConfirmedCases(countryList)

    plot_comparative_GF_ConfirmedCases(countryList)

    plot_comparative_NewFatalities(countryList)

    plot_comparative_GF_Fatalities(countryList)

    

# Plot comparative charts with countries input from user since patient #x:

def comparative_plots_sincePatient(patient, countryList):

    plot_comparative_CC_sincePatient(patient, countryList)

    plot_comparative_F_sincePatient(patient, countryList)

    plot_comparative_CFR_sincePatient(patient, countryList)

    plot_comparative_NewCC_sincePatient(patient, countryList)

    plot_comparative_GrowthCC_sincePatient(patient, countryList)

    plot_comparative_NewF_sincePatient(patient, countryList)

    plot_comparative_GrowthCC_sincePatient(patient, countryList)

    

# Plot maps:

def maps_plot():

    confirmedCases_map()

    new_confirmedCases_map()

    fatalities_map()

    new_fatalities_map()
worlwide_evolution()
maps_plot()
# Manipulating the original dataframe

countrydate_evolution = train[train['ConfirmedCases']>0]

countrydate_evolution = countrydate_evolution.groupby(['Date','Country_Region']).sum().reset_index()



# Creating the visualization

fig = px.choropleth(countrydate_evolution, locations="Country_Region", locationmode = "country names", color="ConfirmedCases", 

                    hover_name="Country_Region", animation_frame="Date", 

                   )



fig.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

    autosize=True,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
perCountry_evolution("Japan")
comparative_plots(["Portugal","Italy", "Spain"])
comparative_plots_sincePatient(100,["Portugal", "Netherlands", "Germany", "Italy", "Spain", "France", "United Kingdom"])