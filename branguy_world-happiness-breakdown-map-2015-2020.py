def create_master(data_2015, data_2016, data_2017, data_2018, data_2019, data_2020):

    

    all_years = data_2020.copy()

    all_years = all_years.rename(columns = {"Ladder score" : "Happiness Score"})

    all_years = all_years.rename(columns = {"Country name" : "Country"})

    

    data_2015 = data_2015.set_index("Country")

    all_years = all_years.merge(data_2015, on = "Country", suffixes = (None, "_2015"))

    

    data_2016 = data_2016.set_index("Country")

    all_years = all_years.merge(data_2016, on = "Country", suffixes = (None, "_2016"))

    

    data_2017 = data_2017.rename(columns = {"Happiness.Score" : "Happiness Score"})

    data_2017 = data_2017.set_index("Country")

    all_years = all_years.merge(data_2017, on = "Country", suffixes = (None, "_2017"))

    

    data_2018 = data_2018.rename(columns = {"Country or region" : "Country"})

    data_2018 = data_2018.rename(columns = {"Score" : "Happiness Score"})

    data_2018 = data_2018.set_index("Country")

    all_years = all_years.merge(data_2018, on = "Country", suffixes = (None, "_2018"))

    

    data_2019 = data_2019.rename(columns = {"Country or region" : "Country"})

    data_2019 = data_2019.rename(columns = {"Score" : "Happiness Score"})

    data_2019 = data_2019.set_index("Country")

    all_years = all_years.merge(data_2019, on = "Country", suffixes = (None, "_2019"))

    

    all_years = all_years.rename(columns = {"Happiness Score" : "Happiness Score_2020"})

    

    return all_years
#Setup

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

data_2015 = pd.read_csv('../input/world-happiness-report/2015.csv')

data_2016 = pd.read_csv('../input/world-happiness-report/2016.csv')

data_2017 = pd.read_csv('../input/world-happiness-report/2017.csv')

data_2018 = pd.read_csv('../input/world-happiness-report/2018.csv')

data_2019 = pd.read_csv('../input/world-happiness-report/2019.csv')

data_2020 = pd.read_csv('../input/world-happiness-report/2020.csv')



all_data = create_master(data_2015, data_2016, data_2017, data_2018, data_2019, data_2020)



#‘YlGnBu’ ‘BuPu’‘OrRd’ 'RdPu’ YlOrRd.

#new = old[['A', 'C', 'D']].copy()





all_data.to_csv("CompiledData.csv")



boundaries_dir = ('../input/python-folio-country-boundaries/world-countries.json')

print("Setup Complete")

all_data
def happy_map(border_dir, curr_data, map1, year):

    #countries1 = pd.read_csv('../input/world-happiness-report/' + year + '.csv')

    

    if "Country or region" in curr_data.columns:

        curr_data = curr_data.rename(columns = {"Country or region" : "Country"})

        

    if "Country name" in curr_data.columns:

        curr_data = curr_data.rename(columns = {"Country name" : "Country"})

        

    if "Score" in curr_data.columns:

        curr_data = curr_data.rename(columns = {"Score" : "Happiness Score"})

        

    if "Happiness.Score" in curr_data.columns:

        curr_data = curr_data.rename(columns = {"Happiness.Score" : "Happiness Score"})

        

    if "Ladder score" in curr_data.columns:

        curr_data = curr_data.rename(columns = {"Ladder score" : "Happiness Score"})

    

    if "United States" in curr_data.Country.tolist():

        curr_data["Country"] = curr_data["Country"].replace({"United States":"United States of America"})







    myscale = (curr_data['Happiness Score'].quantile((0,0.2,0.3,0.4,0.5,0.6,0.8,0.9,0.95,1))).tolist()

    

    folium.Choropleth(

        geo_data = border_dir,

        bins = myscale,

        name = "happiness " + year,

        data = curr_data,

        columns = ["Country", "Happiness Score"],

        key_on = "feature.properties.name",

        fill_color = "RdBu",

        fill_opacity = 1,

        line_opacity = .3,

        #legend_name = "Happiness Score"

    ).add_to(map1)







    return currmap

currmap = folium.Map(location = [0,0], zoom_start = 2)



currmap = happy_map(boundaries_dir, data_2015, currmap, "2015")

currmap = happy_map(boundaries_dir, data_2016, currmap, "2016")

currmap = happy_map(boundaries_dir, data_2017, currmap, "2017")

currmap = happy_map(boundaries_dir, data_2018, currmap, "2018")

currmap = happy_map(boundaries_dir, data_2019, currmap, "2019")

currmap = happy_map(boundaries_dir, data_2020, currmap, "2020")



folium.LayerControl().add_to(currmap)

currmap
countries = pd.read_csv('../input/world-happiness-report/2015.csv')

countries.head()
#Charting top changing countries comparing 2015-2020



copy_2015 = data_2015.copy()

copy_2020 = data_2020.copy()



copy_2015 = copy_2015.set_index("Country")

copy_2015 = copy_2015.sort_index()



copy_2020 = copy_2020.set_index("Country name")

copy_2020 = copy_2020.sort_index()





copy_2020["2015_2020_diff"] = copy_2015["Happiness Score"] - copy_2020["Ladder score"]

copy_2020["2015_score"] = copy_2015["Happiness Score"]

#copy_2015["Happiness Score"].describe()



largest_diff = (copy_2020.nlargest(n = 8,columns = "2015_2020_diff"))["2015_2020_diff"].to_frame()

largest_diff = largest_diff.sort_index()

largest_diff







plt.figure(figsize = (10,6))

plt.title("Countries with largest decrease in Happiness Score 2015->2020")

sns.barplot(data = largest_diff, y = "2015_2020_diff", x = largest_diff.index)

copy_2015 = data_2015.copy()

copy_2015 = copy_2015.set_index("Country").sort_index()



copy_2020 = data_2020.copy()

copy_2020 = copy_2020.set_index("Country name").sort_index()



copy_2020["2015_2020_increase"] = copy_2015["Happiness Score"] - copy_2020["Ladder score"]

copy_2020["2015_score"] = copy_2015["Happiness Score"]



increase_diff = copy_2020.nsmallest(n = 8, columns = "2015_2020_increase")["2015_2020_increase"].to_frame()

increase_diff = increase_diff.sort_index()

increase_diff = increase_diff.abs()



plt.figure(figsize = (14,6))

plt.title("Countries with largest increase from years 2015->2020")

sns.barplot(data = increase_diff, y = "2015_2020_increase", x = increase_diff.index)
copy_2015 = data_2015.copy()

copy_2015 = copy_2015.set_index("Country").sort_index()



copy_2020 = data_2020.copy()

copy_2020 = copy_2020.set_index("Country name").sort_index()



copy_2020["2015_2020_increase"] = copy_2015["Happiness Score"] - copy_2020["Ladder score"]

copy_2020["2015_score"] = copy_2015["Happiness Score"]



copy_2020["abs_score_change"] = copy_2020["2015_2020_increase"].abs()



min_change = copy_2020.nsmallest(n = 8, columns = "abs_score_change")["abs_score_change"].to_frame()

min_change = min_change.sort_index()



plt.figure(figsize = (12, 6))

plt.title("Countries with least change between years 2015 to 2020")

sns.set_style("dark")

sns.barplot(data = min_change, y = "abs_score_change", x = min_change.index)



copy_all = all_data.copy()

copy_all.head()



copy_all["Mean_Happiness_Score"] = (copy_all["Happiness Score_2020"] + copy_all["Happiness Score_2015"] + \

copy_all["Happiness Score_2016"] + copy_all["Happiness Score_2017"] + copy_all["Happiness Score_2018"] + \

copy_all["Happiness Score_2019"]) / 6



copy_all["Variance"] = copy_all[["Happiness Score_2020", "Happiness Score_2015", "Happiness Score_2016", \

                                "Happiness Score_2017", "Happiness Score_2018", "Happiness Score_2019", \

                                ]].var(axis = 1)

copy_all = copy_all.sort_values(by = "Variance", axis = 0)



lowest_var = copy_all.nsmallest(6, "Variance")



lowest_var_trimmed = lowest_var[["Happiness Score_2015", "Happiness Score_2016", \

                                "Happiness Score_2017", "Happiness Score_2018", "Happiness Score_2019", \

                                "Happiness Score_2020"]].copy().set_axis(labels = lowest_var["Country"])



listified_var_trimmed = lowest_var_trimmed.transpose()



plt.figure(figsize = (12, 6))

plt.title("Countries with most steady Happiness Scores")

sns.set_style("dark")

sns.lineplot(data = listified_var_trimmed, hue = "event")

highest_var = copy_all.nlargest(6, "Variance")



highest_var_trimmed = highest_var[["Happiness Score_2015", "Happiness Score_2016", \

                                "Happiness Score_2017", "Happiness Score_2018", "Happiness Score_2019", \

                                "Happiness Score_2020"]].copy().set_axis(labels = highest_var["Country"])



listified_high_var_trimmed = highest_var_trimmed.transpose()





plt.figure(figsize = (12, 6))

plt.title("Countries with most variable Happiness Scores")

sns.set_style("dark")

sns.lineplot(data = listified_high_var_trimmed, hue = "event")
scores = copy_all[["Country", "Happiness Score_2020", "Regional indicator", "Happiness Score_2015", "Happiness Score_2016", \

                                "Happiness Score_2017", "Happiness Score_2018", "Happiness Score_2019", \

                                ]].copy()

scores_mean = scores.groupby("Regional indicator").mean()



#print(df.rename(columns=lambda s: s*3, index=lambda s: s + '!!'))

scores_mean = scores_mean.rename(columns = lambda s: s + " Average")

scores_mean_transpose = scores_mean.transpose()



plt.figure(figsize = (16, 6))

plt.title("Average Happiness by Region")

sns.set_style("whitegrid")

sns.lineplot(data = scores_mean_transpose, hue = "event", dashes = False, sort = True)