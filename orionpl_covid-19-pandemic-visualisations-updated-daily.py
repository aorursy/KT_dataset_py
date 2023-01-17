# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator

import seaborn as sns



# Start Seaborn Templates and read Covid Dataset

sns.set()

sns.set_style("whitegrid")



dataset = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

dataset.tail()
dataset.info()
# Drop unnecessary collums

dataset.drop(labels=["SNo", "Last Update"], axis=1, inplace=True)



# And set up "Unkown" value for Coutries without specified states

dataset.fillna(value="Unknown", inplace=True)



# Some name fixes

dataset.rename(columns={"Country/Region": "Country", "Province/State": "Province"}, inplace=True)



dataset.loc[ dataset["Country"] == "Mainland China", "Country"] = "China"

dataset.loc[ dataset["Country"] == "Others", "Country"] = "Diamond Princess"

dataset.loc[ dataset["Province"] == "Bavaria", "Province"] = "Unknown"

dataset.loc[ dataset["Province"] == "None", "Province"] = "Unknown"

dataset.loc[ dataset["Province"] == "From Diamond Princess", "Province"] = "Unknown"



# Create new column for currently affected patients

dataset["Affected"] = dataset["Confirmed"] - dataset["Deaths"] - dataset["Recovered"]

dataset.info()

# Some dates have year as 2020 and some as 20, so we want to standardized that

def clean_date(date):

    if len(date) > 8:

        return date[0:8]

    else:

        return date



dataset["ObservationDate"] = dataset["ObservationDate"].apply(clean_date)
# Now we read data set with country and continets data, get ability to compare how every continent is doing 

cou_cont = pd.read_csv("../input/country-to-continent/countryContinent.csv", encoding="iso-8859-1")



# Drop unnecessary colums, rename to match covid dateset and add some missing countries

cou_cont = cou_cont[["country", "continent"]]

cou_cont.rename(columns={"country": "Country", "continent": "Continent"}, inplace=True)

more_continents = pd.DataFrame([

    ["Macau", "Asia"], 

    ["Ivory Coast", "Africa"], 

    ["North Ireland", "Europe"], 

    ["North Macedonia", "Europe"], 

    ["UK", "Europe"], 

    ["Iran", "Asia"], 

    [" Azerbaijan", "Asia"], 

    ["Others", "Other"], 

    ["Russia", "Europe"], 

    ["Taiwan", "Asia"], 

    ["US", "Americas"], 

    ["South Korea", "Asia"], 

    ["Vietnam", "Asia"],

    ['Diamond Princess', 'Asia'], 

    ['Saint Barthelemy', 'Americas'],

    ['Palestine', 'Asia'],

    ['Vatican City', 'Europe'],

    ['Republic of Ireland', 'Europe'], 

    ['Moldova', 'Europe'],

    ['St. Martin', 'Americas'],

    ['Brunei', 'Asia'],

    ['occupied Palestinian territory',  'Asia'],

    ["('St. Martin',)", 'Americas'],

    ['Channel Islands', 'Europe'],

    ['Bolivia',  'Americas'],

    ['Congo (Kinshasa)',  'Africa'],

    ['Reunion', 'Africa'],

    ['Venezuela',  'Americas'],

    ['Curacao',  'Americas'],

    ['Eswatini', 'Africa'],

    ['Kosovo', 'Europe'],

    ['Congo (Brazzaville)',  'Africa'],

    ['Republic of the Congo',  'Africa'],

    ['Tanzania', 'Africa'],

    ['The Bahamas',  'Americas'],

    ['The Gambia',  'Africa'],

    ['Gambia, The',  'Africa'],

    ['Bahamas, The', 'Americas'],

    ['Cape Verde', 'Africa'],

    ['East Timor',  'Oceania'],

    ['Syria',  'Asia'],

    ['Laos',  'Asia'],

    ['West Bank and Gaza', 'Asia'],

    ['Burma',  'Asia'],

    ['MS Zaandam', 'Americas'],

], columns=["Country", "Continent"])



cou_cont = cou_cont.append(more_continents)



# Now merge 2 datasets, and check if its all right

dataset = pd.merge(dataset, cou_cont, on="Country", how="left")



dataset.info()

#dataset.loc[ dataset.isnull().any(axis=1), "Country"].unique()
# Create some grouped datasets for later use

latest = dataset.groupby(["Province", "Country"]).last().reset_index().copy()

first = dataset.loc[dataset["Confirmed"] > 0, ["ObservationDate", "Country", "Confirmed"]].groupby(["Country"]).first().reset_index().copy()

global_daybyday = dataset.groupby("ObservationDate").sum().reset_index().copy()



# Create color maps

cmap = plt.get_cmap("tab20")

tab20b = plt.get_cmap("tab20b")



colors_dict = {

    "Confirmed": cmap(2),

    "Affected": cmap(0),

    "Deaths": cmap(6),

    "DeathRatio": cmap(6),

    "Recovered": cmap(4),

}



colors_cont = {

    "Europe": cmap(0),

    "Asia": cmap(6),

    "Americas": cmap(4),

    "Oceania": cmap(19),

    "Africa": cmap(2),

}



colors = cmap(np.array([2, 4, 6, 0, 19, 15]))

#print(colors_dict)



# And some chart variables

xtl_size = 14

ytl_size = 14

date_step = 4
# We sum number of confirmed cases in every continent in certain day

cont_daybyday = dataset.groupby(["ObservationDate", "Continent"]).sum().reset_index().copy()



# Change it to long df format

cont_long = pd.melt(cont_daybyday[["ObservationDate", "Confirmed", "Continent"]], id_vars=["ObservationDate", "Continent"], value_vars=["Confirmed"])

                   

# And draw chart

plt.figure(figsize=(15,5))



ax = sns.lineplot(x="ObservationDate", y="value", hue="Continent", data=cont_long, palette=colors_cont)

sns.despine()



plt.xticks(rotation='vertical')



ax.spines['left'].set_color('none')



ax.set_ylim(0, cont_daybyday["Confirmed"].max()*1.1)

ax.set_xlabel("")

ax.set_ylabel("")



ax.tick_params(axis="y", labelsize=ytl_size)

ax.tick_params(axis="x", labelsize=xtl_size)



ax.set_xticks(ax.get_xticks()[::date_step])



ax.set_title("Confirmed Cases by Continent day by day", fontsize=20)



ax.legend(labels=cont_long["Continent"].unique(), facecolor='white', edgecolor='white', prop={'size': xtl_size})

ax.grid(False, axis="x")



plt.show()
# So now we count present number of currently affected and confirmed cases for every continent

continents = latest.groupby(["Continent"]).sum().reset_index().copy()



# And plot it as bar plot

fig, axes = plt.subplots(1, 2)

fig.set_size_inches(11,5)





axes[0].pie(continents["Confirmed"], startangle=90, radius=1, colors=colors, wedgeprops=dict(width=0.3, edgecolor='w'))

axes[0].set_title('Confirmed cases per continent', fontsize=17)



axes[1].pie(continents["Affected"], startangle=90, radius=1, colors=colors, wedgeprops=dict(width=0.3, edgecolor='w'))

axes[1].set_title('Currently Affected per continent', fontsize=17)



handles = axes[1].get_legend_handles_labels()



fig.legend(labels = continents["Continent"].unique(), loc='center', frameon=False)



plt.show()

#continents.sort_values("Confirmed", ascending=False).reset_index(drop=True).head(10)
# We sum number of cases in every country to get number of cases worldwide

daybyday = pd.melt(global_daybyday[["ObservationDate", "Confirmed","Deaths", "Affected", "Recovered"]], id_vars=["ObservationDate"], value_vars=["Confirmed","Deaths", "Affected", "Recovered"])



# And plot it as line plot

plt.figure(figsize=(15,5))



ax = sns.lineplot(x="ObservationDate", y="value", hue="variable", data=daybyday, palette=colors_dict)



sns.despine()



plt.xticks(rotation='vertical')



ax.spines['left'].set_color('none')



ax.set_ylim(0, global_daybyday["Confirmed"].max()*1.1)

ax.set_xlabel("")

ax.set_ylabel("")



ax.tick_params(axis="y", labelsize=ytl_size)

ax.tick_params(axis="x", labelsize=xtl_size)



ax.set_xticks(ax.get_xticks()[::date_step])



ax.set_title("Global spread of pandemic", fontsize=20)

ax.legend(labels=["Confirmed","Deaths", "Affected", "Recovered"], facecolor='white', edgecolor='white', prop={'size': xtl_size})

ax.grid(False, axis="x")



plt.show()
# In first step of panthemic most cases was raported in china, later USA become country with higher number

# of cases, we want to show number of new cases every day in China, USA, Rest of world and Total, to compare

# different states od pancemic



# First we count number of cases worldwide every day, and take derivative to get number of new cases

world_new = dataset.groupby("ObservationDate").sum().reset_index().set_index("ObservationDate").copy()

world_new = world_new["Confirmed"].diff().dropna()



# Now we do the same but first filter data to China only

china_new = dataset[ dataset["Country"] == "China" ].groupby("ObservationDate").sum().reset_index().set_index("ObservationDate").copy()

china_new = china_new["Confirmed"].diff().dropna()



# And US only

us_new = dataset[ dataset["Country"] == "US" ].groupby("ObservationDate").sum().reset_index().set_index("ObservationDate").copy()

us_new = us_new["Confirmed"].diff().dropna()



# And now rest of the world

nonchina_new = dataset[ (dataset["Country"] != "China") & (dataset["Country"] != "US") ].groupby("ObservationDate").sum().reset_index().set_index("ObservationDate").copy()

nonchina_new = nonchina_new["Confirmed"].diff().dropna()



# Now we make one dateframe from all 4 above in long format

new_cases = pd.concat({'Total': world_new, 'China': china_new, 'Non US, non China': nonchina_new, 'USA': us_new}).reset_index()



# And plot it as line plot

plt.figure(figsize=(15,5))



ax = sns.lineplot(x="ObservationDate", y="Confirmed", hue="level_0", data=new_cases, palette={'USA': cmap(0), 'China': cmap(6), 'Non US, non China': cmap(4), 'Total': cmap(2)})

sns.despine()



plt.xticks(rotation='vertical')



ax.spines['left'].set_color('none')



ax.set_ylim(0, new_cases["Confirmed"].max()*1.1)

ax.set_xlabel("")

ax.set_ylabel("")



ax.tick_params(axis="y", labelsize=ytl_size)

ax.tick_params(axis="x", labelsize=xtl_size)



ax.set_xticks(ax.get_xticks()[::date_step])



ax.set_title("New confirmed cases day by day", fontsize=20)



ax.legend(labels=new_cases["level_0"].unique(), facecolor='white', edgecolor='white', prop={'size': xtl_size})

ax.grid(False, axis="x")



plt.show()
# We want to plot situation in countries with most cases so we use latest df, that have present data,

# and sum by country (to get rid of states), and get top10 values

top_10 = latest.groupby("Country").sum().sort_values("Confirmed", ascending=False).reset_index().head(10).copy()



# Drop unnecesary columns and change df format to long

top_10_wide = pd.melt(top_10[["Country", "Recovered", "Deaths", "Affected"]], id_vars=["Country"], value_vars=["Affected", "Recovered","Deaths"])



# And plot it as combined bar plot

plt.figure(figsize=(15,8))

ax = sns.barplot(y="Country", x="value", hue="variable", data=top_10_wide, palette=colors_dict)



ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.spines['bottom'].set_color('none')



ax.tick_params(axis="y", labelsize=ytl_size)

ax.tick_params(axis="x", labelsize=xtl_size)



ax.set_xlabel("")

ax.set_ylabel("")

ax.set_title("Countries with highest number of cases", fontsize=20)



ax.legend(facecolor='white', edgecolor='white', prop={'size': xtl_size}).set_title('')



plt.show()
# Now, we want to show current highest grows as barplot, so we group df by country and date

confirmed_wide = dataset[["ObservationDate", "Confirmed", "Province", "Country"]].groupby(["Country", "ObservationDate"]).sum().reset_index().copy()



# Change format to wide, to have different column for every country, and fill missing values with 0.0

confirmed_wide = pd.pivot(confirmed_wide, columns="Country", index="ObservationDate", values="Confirmed").fillna(0.0)



# Take derivative to get number of new cases, drop other than last rows, sort by highest grow, and limit to 10 countries

confirmed_wide = confirmed_wide.diff().transpose().iloc[:,-1].sort_values(ascending=False).head(10)



# and now we plot it as barplow

plt.figure(figsize=(15,8))

ax = sns.barplot(x=confirmed_wide.values, y=confirmed_wide.index, palette=sns.color_palette("GnBu_d", 10))



ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.spines['bottom'].set_color('none')



ax.tick_params(axis="x", labelsize=xtl_size)

ax.tick_params(axis="y", labelsize=ytl_size)



ax.set_xlabel("")

ax.set_ylabel("")

ax.set_title("Yesterday new cases by country", fontsize=20)



ax.legend(frameon=False).set_title('')



plt.show()
# country_praphs function, that can show us situation in given country

# country - name of country, in str format

# state - False if we dont want the line plot of Confirmed/Affected/Recovered/Deaths

# rise - False if we dont want to combined barplot of new cases and line plot of cumulative cases

def country_graphs(country, state=True, rise=True):

    

    # get from ds data about give country

    cdataset = dataset[ dataset["Country"] == country ].groupby("ObservationDate").sum().reset_index().copy()

    

    # Line plot of Confirmed/Affected/Recovered/Deaths

    if state == True:

        

        # Change format of data to long format needed by matplotlib

        cdataset_long = pd.melt(cdataset[["ObservationDate", "Confirmed", "Deaths", "Affected", "Recovered"]], id_vars=["ObservationDate"], value_vars=["Confirmed","Deaths", "Affected", "Recovered"])



        # And plot it as line chart

        plt.figure(figsize=(15,5))

        

        ax = sns.lineplot(x="ObservationDate", y="value", hue="variable", data=cdataset_long, palette=colors_dict)

        sns.despine()

        

        plt.xticks(rotation='vertical')

        

        ax.set_ylim(0, cdataset["Confirmed"].max()*1.1)

        ax.spines['left'].set_color('none')

        

        ax.set_xlabel("")

        ax.set_ylabel("")

        

        ax.tick_params(axis="y", labelsize=ytl_size)

        ax.tick_params(axis="x", labelsize=xtl_size)

        ax.set_xticks(ax.get_xticks()[::date_step])

        

        ax.set_title("Spread of pandemic in " + country, fontsize=20)

        ax.legend(labels=["Confirmed","Deaths", "Affected", "Recovered"], facecolor='white', edgecolor='white', prop={'size': xtl_size})

        ax.grid(False, axis="x")

    

    # Combined barplot of new cases and line plot of cumulative cases

    if rise == True:

        

        # Copy data from df but only date and number confirmed cases

        cconfirmed = cdataset[["Confirmed", "ObservationDate"]].copy()



        # Take derivative of number cases to get number of new cases

        cconfirmed["New"] = cconfirmed["Confirmed"].diff()

        

        # Plot bar plot as ax1, and line plot as ax2

        fig = plt.figure(figsize=(15,5))

     

        ax1 = fig.add_subplot(111)

        g1 = sns.barplot(x="ObservationDate", y="New", data=cconfirmed, color=colors_cont["Oceania"], ax=ax1)



        plt.xticks(rotation='vertical')    



        ax2 = ax1.twinx()

        g2 = sns.lineplot(x="ObservationDate", y="Confirmed", data=cconfirmed, color=colors_dict["Confirmed"], ax=ax2)

        

        sns.despine()

        

        ax1.set_ylim(0)

        ax1.grid(False)

        ax1.tick_params(axis='y', labelcolor=colors_dict["Affected"], labelleft=False, labelright=True, left=False, right=True) 

   

        ax2.set_ylim(0)

        ax2.grid(False, axis="x")

        ax2.tick_params(axis='y', labelcolor=colors_dict["Confirmed"], labelleft=True, labelright=False, left=True, right=False) 



        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

        ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax2.get_yticks())))        



        ax1.tick_params(axis="y", labelsize=ytl_size)

        ax2.tick_params(axis="y", labelsize=ytl_size)

        ax1.tick_params(axis="x", labelsize=xtl_size+2)

        ax2.set_xticks(ax2.get_xticks()[::date_step])

        

        ax1.spines['left'].set_color('none')

        ax1.set_xlabel("")

        ax1.set_ylabel("")

        ax2.set_ylabel("")        

        ax1.set_title("Cumulative and New cases in " + country, fontsize=20)

        

        handles = [Line2D([0], [0], marker='o', color=colors_dict["Confirmed"]),

                   Patch(color=colors_cont["Oceania"])]

        

        ax2.legend(handles = handles, labels = ["Confirmed", "New"], facecolor='white', edgecolor='white', loc=2, prop={'size': xtl_size})



        plt.show()



# Italy example

country_graphs("Italy", True, True)
# Now, we want to show change in covid mortality rate

# To count this, we need number of confirmed cases, and deaths every day

mortality = global_daybyday[["ObservationDate", "Confirmed", "Deaths"]].copy()



# Mortality rate is deaths/confirmed, *100 to get percentage

mortality["DeathRatio"] = mortality["Deaths"] * 100 / mortality["Confirmed"]



# Now we plot change as line plot

plt.figure(figsize=(15,5))

ax = sns.lineplot(x="ObservationDate", y="DeathRatio", data=mortality, palette=colors_dict)

sns.despine()



ax.set_ylim(0, mortality["DeathRatio"].max()*1.1)



ax.set_xlabel("")

ax.set_ylabel("")



plt.xticks(rotation='vertical')

ax.tick_params(axis="y", labelsize=ytl_size)

ax.tick_params(axis="x", labelsize=xtl_size)

ax.set_xticks(ax.get_xticks()[::date_step])

        

ax.spines['left'].set_color('none')

ax.set_title("Change of Global Mortality Rate in time", fontsize=20)

ax.grid(False, axis="x")



# And add arrow with text, to easly show current value

plt.annotate(str(round(mortality["DeathRatio"].iloc[-1],2)) + "%", 

            xy=(mortality.shape[0]-1, mortality["DeathRatio"].iloc[-1]),  

            xycoords='data',

            xytext=(mortality.shape[0]-5, mortality["DeathRatio"].iloc[-1]-0.8), 

            textcoords='data',

            arrowprops=dict(color=('black'),

                            arrowstyle="<-",

                            connectionstyle="arc3"))



plt.show()
# Now we want to compare mortality rate in countries with highest number of cases

# Get dataset with present data, and get rid of states

sickest_countries = latest.groupby("Country").sum().reset_index().copy()



# Limit df to only countries with 100000+ cases

sickest_countries = sickest_countries[sickest_countries["Confirmed"] > 100000][["Country", "Confirmed", "Deaths"]]



# Add World row

sickest_countries = sickest_countries.append({"Country":"World", 

                                              "Confirmed": latest["Confirmed"].sum(), 

                                              "Deaths": latest["Deaths"].sum()}, 

                                             ignore_index=True)



# Calculate mortality rate, and sort from hightest to lowest

sickest_countries["DeathRatio"] = round(sickest_countries["Deaths"] * 100 / sickest_countries["Confirmed"], 2)



sickest_countries = sickest_countries.sort_values("DeathRatio", ascending=False).reset_index(drop=True)



# Color palette

mort_colors = sns.dark_palette("purple", sickest_countries.shape[0])



mort_colors[sickest_countries[ sickest_countries["Country"] == "World" ].index[0]] = tab20b(0)



# Bar plot

plt.figure(figsize=(15,8))

ax = sns.barplot(x="DeathRatio", y="Country", data=sickest_countries, palette=mort_colors)



ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.spines['bottom'].set_color('none')



ax.tick_params(axis="y", labelsize=ytl_size)

ax.tick_params(axis="x", labelsize=xtl_size)



ax.set_xticks(ax.get_xticks()[::date_step])



# Text

for i in range(0, sickest_countries.shape[0]):

    ax.text(sickest_countries.iloc[i,3]+0.1,

        i+0.1,

        str(round(sickest_countries.iloc[i,3],2)) + "%", 

        color='black', ha="left", fontsize=14, backgroundcolor="white")

 

ax.set_xlabel("")

ax.set_ylabel("")

ax.set_title("Mortality Rate (in terms of Confirmed) in Countries with 100000+ cases", fontsize=20)



ax.legend(frameon=False).set_title('')



plt.show()
# For last plot, we want to show top 10 most deadliest days (counting every country separately)

# To do this, we need to sum dateset by country (to get rid of states)

deadliest_days = dataset[["ObservationDate", "Deaths", "Province", "Country"]].groupby(["Country", "ObservationDate"]).sum().reset_index().copy()



# Now we change format to wide, becouse we need separete column for every country, and take derivatice

# to get number of new deaths in certain day

deadliest_days = pd.pivot(deadliest_days, columns="Country", index="ObservationDate", values="Deaths").fillna(0.0).diff().fillna(0.0).reset_index()



# Now we change format to long, and take top 10 days

deadliest_long = pd.melt(deadliest_days, id_vars=["ObservationDate"], value_vars=deadliest_days.columns.to_list()[1:]).sort_values("value", ascending=False).reset_index(drop=True).head(10)

 

# And plot date using bar plot

plt.figure(figsize=(15,8))

ax = sns.barplot(x=deadliest_long["value"], y=deadliest_long.index.to_list(), palette=sns.dark_palette("purple", 10), orient="h")



ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.spines['bottom'].set_color('none')



# Add text to bars

for i in range(0, 10):

    ax.text(25,

        i+0.1,

        deadliest_long.iloc[i,0] + ' - ' + deadliest_long.iloc[i,1] + ' - ' + str(int(deadliest_long.iloc[i,2])) + "†", 

        color='white', ha="left", fontsize=14)

 

ax.set_xlabel("")

ax.set_ylabel("")

ax.set_title("Higheest numbers of new deaths in one country", fontsize=20)



plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)



ax.legend(frameon=False).set_title('')

ax.grid(False)



plt.show()