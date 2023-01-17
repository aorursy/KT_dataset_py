import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
d215 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
d216 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")
d217 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")
d218 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
d219 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

dlist = [d215, d216, d217, d218, d219]
years = [2015, 2016, 2017, 2018, 2019]

for x, y in zip(years,dlist):
    print("Shape of data from ", x, ": ", y.shape, "\n")
for x in dlist:
    print(x.columns)
d217
def compare_columns(df1, df2):
    """ 
    Takes two lists or tuples as arguments and compares them
    
    We can use this function to compare two lists and get the values that are in one list and not in the other
    """
    
    c1 = df1.columns
    c2 = df2.columns
    
    rl1 = list()
    
    for value in c1:
        if value not in c2:
            rl1.append(value)
            
    print("Comparing Values")        
            
    for x in rl1:
        print(x, "not in second dataframe")
        
compare_columns(d215,d218)
        
# Renaming Columns due to data inconsistency
d215_1 = d215.rename(columns = {"Happiness Rank":"Rank", "Happiness Score":"Score", "Economy (GDP per Capita)":"GDP",
                             "Health (Life Expectancy)":"Life Expectancy", "Trust (Government Corruption)":"Trust"})
d216_1 = d216.rename(columns = {"Happiness Rank": "Rank", "Happiness Score": "Score", "Lower Confidence Interval":"Lower",
                             "Upper Confidence Interval":"Upper", "Economy (GDP per Capita)":"GDP", 
                              "Health (Life Expectancy)":"Life Expectancy", "Trust (Government Corruption)":"Trust"})
d217_1 = d217.rename(columns = {"Happines.Rank":"Rank", "Happiness.Score":"Score", "Happiness.Rank":"Happiness", 
                              "Whisker.high":"High", "Whisker.low":"Low", "Economy..GDP.per.Capita.":"GDP", 
                              "Health..Life.Expectancy.":"Life Expectancy", "Trust..Government.Corruption.":"Trust",
                             "Dystopia.Residual":"Dystopia Residual"})
d218_1 = d218.rename(columns = {"Country or region":"Country", 
                              "Freedom to make life choices":"Freedom", 
                              "GDP per capita":"GDP", 
                              "Overall rank":"Rank", 
                              "Social support":"Social Support", 
                              "Perceptions of corruption":"Trust",
                             "Healthy life expectancy":"Life Expectancy"})
d219_1 = d219_1.rename(columns = {"Overall rank":"Rank", "Country or region": "Country", "GDP per capita":"GDP", "Social support":"Social Support",
                             "Healthy life expectancy":"Life Expectancy", "Freedom to make life choices":"Freedom", "Perceptions of corruption":"Trust"})


# d215 = d215.drop("Standard Error", axis = 1)

dlist = [d215_1, d216_1,d217_1,d218_1,d219_1]

d219_1
# Creting a function that averages that values of a column in a table

def get_yearly_averages(value, dataframes):
    """
    Gets averages for certain values from each year
    
    Value - the Value as a string of the column that should be averaged for each year
    Dataframes - takes a list of dataframes
    """
    averages = list()
    
    for x in dataframes:
        averages.append(x[value].mean())
        
    averages = list(zip(years,averages))                                  # Years referenced in earlier cell
    averages = pd.DataFrame(averages, columns = ["Year", value])
    averages = averages.set_index("Year")                                 # Making sure we index by year, not 1,2,3,..
    
    return averages

# Average Happiness score Dataframe (Global)
avg_score = get_yearly_averages("Score", dlist)

# Average GDP (Global)
avg_gdp = get_yearly_averages("GDP", dlist)

# Average Life Expectancy (Global)
avg_le = get_yearly_averages("Life Expectancy", dlist)


yearly_averages = pd.concat([avg_score, avg_gdp, avg_le], sort=True, axis = 1)

avg_le.plot(kind="bar")
import matplotlib.pyplot as plt

hap_15_avg = d215["Score"].mean()
hap_16_avg = d216["Score"].mean()
hap_17_avg = d217["Score"].mean()
hap_18_avg = d218["Score"].mean()
hap_19_avg = d219["Score"].mean()

data = [[2015, hap_15_avg], [2016, hap_16_avg], [2017, hap_17_avg], [2018, hap_18_avg], [2019, hap_19_avg]]
avg_hap = pd.DataFrame(data, columns = ["Year", "Happiness"])
avg_hap.astype({"Year":"int"})
avg_hap = avg_hap.set_index("Year")
avg_hap.plot(kind="line", alpha = 0.5)
plt.xticks(range(2015,2020))
plt.title("Average Happiness over the years")
d215_2 = d215_1.set_index("Country")
d216_2 = d216_1.set_index("Country")
d217_2 = d217_1.set_index("Country")
d218_2 = d218_1.set_index("Country")
d219_2 = d218_1.set_index("Country")

s16 = d216_2.loc["Syria"]
s17 = d217_2.loc["Syria"]
s18 = d218_2.loc["Syria"]
s19 = d219_2.loc["Syria"]

s16