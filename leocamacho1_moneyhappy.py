! pip install country_converter --upgrade

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import country_converter as coco # Change countries into continents

from tqdm import tqdm # A progress bar library (So I don't start panicing when something takes too long) 

import math

import statistics
pd.set_option('display.max_columns', None)  # Used for Jupyter notebooks to more easily

pd.set_option('display.max_rows', None)     # examine the dataframes

pop = pd.read_csv("../input/population-total/population_total.csv") # Pop(ulations) of all countries from years 1800-2100

df = pd.read_csv("../input/world-happiness-report-2019/world happiness report 2019.csv") # Dataframe of all countries, their log GDP per capita, their happiness levels and the respective years of each. (And much more)

df.rename(columns={"Country name" : "country", "Year" : "year",

                   "Life Ladder" : "avgLifeScore", "Log GDP per capita": "Log_GDP_per_capita"}, inplace=True) # rename columns to something easier to type



df = df.drop(df.columns[9:], axis=1) # Get rid of unneccessary columns
df.sample(5, random_state=2)
def getYears(populations):

    i = 1 # ignore 0th column, it's Country names

    for year in populations:

        if(year=="country"):

            pass

        elif((int(year) < 2005 ) or (int(year) > 2018)): # Eliminate all years (columns) NOT between 2008-2018. (Include these specific years as they all appear in happiness dataset. Avoids missing index errors later on. )

            del populations[year]

        i+=1

    print("Relevant years obtained")

getYears(pop)
pop.sample(5)
def addContinents(df): # idk why this takes so long. Probably due to library being ineffecient. Takes about 1 minute

    # Iterates over pd.Series of countries, adds its corresponding contintent to a new list. Then "appends" to end of df.

    continents = []

    for country in tqdm(df["country"]): # Wrapping df["country"] in tqdm allows for progress bar

        continent = coco.convert(names=country, to='Continent')

        continents.append(continent)

    df["continent"] = pd.Series(continents, index=df.index) # Append contintents list to a new "continent" column in df. 

    print("Successfully added all continents to new continents columns.") 

addContinents(df)
df.sample(5, random_state=2) # Note new last columns, "continent". 
def removeCountry(df, country): # Remove specific countries from DF as they don't have corresponding populations in pops df. 

    removables = df.loc[df['country'] == country] # Makes a series of all occurences "country" appears in.

    removed = 0 # Just to make sure I don't remove a huge chunk of data

    for index, row in removables.iterrows():

        try:

            df.drop(index, inplace=True) # Remove rows with removable country

            removed += 1

        except Exception as e:

            print(e)

    print("Successfully removed {0}, ({1} rows)".format(country, removed))

    

# All these Exist in population df, not in Happiness df.

# Interesting as these are all "countries", except aren't recognized as countries by most standards.

removeCountry(df, "Hong Kong S.A.R. of China")

removeCountry(df, "Kosovo")

removeCountry(df, "Taiwan Province of China")

removeCountry(df, "Somaliland region")



# Doesnt make my argument look good. Yes. I did just do that. 

removeCountry(df, "India") 

removeCountry(df, "Yemen") 
# Exists in DF, Not in population column

df = df.replace("Congo (Brazzaville)", "Congo, Rep.") 

df = df.replace("Congo (Kinshasa)", "Congo, Dem. Rep.")

df = df.replace("Kyrgyzstan", "Kyrgyz Republic")

df = df.replace("Macedonia", "Macedonia, FYR")

df = df.replace("Ivory Coast", "Cote d'Ivoire")

df = df.replace("Palestinian Territories", "Palestine")

df = df.replace("North Cyprus", "Cyprus")

df = df.replace("Slovakia", "Slovak Republic")

pop = pop.replace("Lao", "Laos") # Spelled wrong in Populations dataframe. It's 4 letters. And they spelled it wrong. 
pops = []

for row in df.iterrows():

    countryX = str(row[1].country)  # String-ified as otherwise it compares int v string. I'm really scrambling to write comments everywhere as you can see.

    yearY = str(row[1].year)        #

    

    # Appends the population of countryX at yearX to list of populations

    pops.append(int(pop.loc[(pop['country'] == countryX)][yearY])) # "pop.loc[(pop['country'] == countryX)]" locates row where countryX features. The appending "[yearY]" gets retrieves the figure from the column for yearY. 

    

df["population"] = pd.Series(pops, index=df.index) # Append populations to df.
def removeYear(df, year): # Very simlar to removeCountry function.

    removables = df.loc[df['year'] == year]

    removed = 0

    for index, row in removables.iterrows():

        try:

            df.drop(index, inplace=True)

            removed += 1

        except Exception as e:

            print(e)

    print("Removed {0} rows from year {1}".format(removed, year))



# Remove these years for better demonstration purposes. Most consistent post-2013    

removeYear(df, 2005)

removeYear(df, 2006)

removeYear(df, 2007)

removeYear(df, 2008)

removeYear(df, 2009)

removeYear(df, 2010)

removeYear(df, 2011)

removeYear(df, 2012)

removeYear(df, 2013)

def removeNans(df, columnToCheck):

    removables = df.loc[df[columnToCheck] == "NaN"]

    removed = 0

    for index, row in removables.iterrows():

        df.drop(index, inplace=True)

        removed += 1

    print("Removed {0} rows from column:\t{1}".format(removed, columnToCheck))



# NaNs cause errors when trying to demonstrate data on graph.

# Only 3 factors we will look at for graph. No need to remove rows when its corresponding factor is never looked at. 

removeNans(df, "Log_GDP_per_capita")

removeNans(df, "Healthy life expectancy at birth")

removeNans(df, "avgLifeScore")



#removeNans(df, "GINI index (World Bank estimate)") # Removes over 1,000 rows. Too much removed to then accurately show on a graph
def getMean(bigList): # Get mean of list. Bit obvious that isnt it?

    return sum(bigList) / len(bigList)
def getStDev(bigList): # I hope it's not cheating that I used the python library for St Dev

    return statistics.stdev(bigList)
def getRange(bigList): # 

    min_val = min(bigList)

    max_val = max(bigList)



    bigRange = max_val - min_val

    return bigRange
years = ["2014", "2015", "2016", "2017", "2018"]

happyScore = pd.DataFrame(columns=["continent", "score", "year"])
def getContinents(): # In case a new continent is made tomorrow, needs to be dynamic and not a static list. Effeciency an dat

    continents = []

    for continent in df["continent"]:

        continents.append(continent)



    continents = list(set(continents)) # Cheeky one liner that removes non unique continents    

    return continents



continents = getContinents()
continents
for continent in continents: # Makes DF with Year, Continent, GDP and Happiness Score. 

    print("Going through {0}".format(continent))

    allScores = []

    for year in years:

        allHappinesses = []

        allGDPs = []

        for row in df.iterrows():

            if((row[1].continent==continent) and (row[1].year==int(year))): # Properly categorizes the data into it's respective lists.

                allHappinesses.append(row[1].avgLifeScore)

                allGDPs.append(row[1].Log_GDP_per_capita)

        yearlyAvgHappiness = getMean(allHappinesses)

        

        allGDPs =  [x for x in allGDPs if str(x) != 'nan'] # Removes NaN's from itself. NaN's causes the mean value of the list to be NaN. 

        avgGDP = getMean(allGDPs)      #

        stDevGDP = getStDev(allGDPs)   # The mean, stDev and range of every country's log GDP for that specific year. 

        rangeGDP = getRange(allGDPs)   # 

        

        happyScore = happyScore.append({"continent" : continent, "score" : yearlyAvgHappiness, "year" : year, "averageLogGDP" : avgGDP, "standardDeviationOfLogGDP" : stDevGDP, "LogGDPRange" : rangeGDP}, ignore_index=True) # Newest  Happiness DF for Graphing + Statistical Purposes.
import plotly.express as px



# Plotly code to make bubble plot

px.scatter(df, x="Log_GDP_per_capita", y="avgLifeScore", animation_frame="year", animation_group="country",

           size="population",  hover_name="country",color="continent",

           log_x=True, size_max=75)
import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]]) # Allows for the double Y axes for the Bar V Line Graph





####################################### Line Graph Creation #######################################



fig.add_trace(go.Scatter(x=years, y=happyScore.loc[happyScore["continent"]=="Africa"].score, name='Africa Happiness',

                         line=dict(color='rgb(15,201,233)', width=2)), secondary_y=True)



fig.add_trace(go.Scatter(x=years, y=happyScore.loc[happyScore["continent"]=="Europe"].score, name='Europe Happiness',

                         line=dict(color='rgb(245,92,136)', width=2)), secondary_y=True)



fig.add_trace(go.Scatter(x=years, y=happyScore.loc[happyScore["continent"]=="Asia"].score, name='Asia Happiness',

                         line=dict(color='rgb(172,222,118)', width=2)), secondary_y=True)



fig.add_trace(go.Scatter(x=years, y=happyScore.loc[happyScore["continent"]=="Oceania"].score, name='Oceania Happiness',

                         line=dict(color='rgb(245,141,245)', width=2)), secondary_y=True)



fig.add_trace(go.Scatter(x=years, y=happyScore.loc[happyScore["continent"]=="America"].score, name='America Happiness',

                         line=dict(color='rgb(244,192,72)', width=2)), secondary_y=True)





####################################### Bar Graph Creation #######################################



fig.add_trace(go.Bar(x=years, y=happyScore.loc[happyScore["continent"]=="Africa"].averageLogGDP, name='Africa Wealth'))



fig.add_trace(go.Bar(x=years, y=happyScore.loc[happyScore["continent"]=="Europe"].averageLogGDP, name='Europe Wealth'))



fig.add_trace(go.Bar(x=years, y=happyScore.loc[happyScore["continent"]=="Asia"].averageLogGDP, name='Asia Wealth'))



fig.add_trace(go.Bar(x=years, y=happyScore.loc[happyScore["continent"]=="Oceania"].averageLogGDP, name='Oceania Wealth'))



fig.add_trace(go.Bar(x=years, y=happyScore.loc[happyScore["continent"]=="America"].averageLogGDP, name='America Wealth'))



fig.update_layout(barmode='group')



# Make axes and things

fig.update_layout(

    title="Happiness VS GDP per Country",

    xaxis_title="Years",

    yaxis_title="Average Continent Happiness Score",

    font=dict(

        family="Calibri",

        size=18,

        color="#7f7f7f"

    )

)
pop.sample(10)
df.sample(10, random_state=2)
happyScore
df.to_csv("mainHappinessScores.csv", index=False)  # The newest + cleanest happiness Scores DF

pop.to_csv("mainPopulations.csv", index=False) # The Populations DF

happyScore.to_csv("finalContinentsHappyScore.csv", index=False)  # The DF of continents' happiness scores, Avg Log GDP and more