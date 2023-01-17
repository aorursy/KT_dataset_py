#Import all libraries 

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import missingno as msno
#Import the dataset

world = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")
#View head of the dataset

world.head()
#View non-null count and data types of all columns

world.info()
#Visualize missing values

msno.matrix(world)

plt.show()
#Trim 'Country' column values of whitespaces

world['Country'] = world['Country'].str.strip()

#Trim 'Region' column values of whitespaces

world['Region'] = world['Region'].str.strip()
#All the unique countries in the dataset

world['Country'].unique()
#All the unique regions in the dataset

world['Region'].unique()
#Wrangle columns to replace ',' by '.' and convert to 'float64' datatype

cols = ['Pop. Density (per sq. mi.)','Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)','Literacy (%)','Phones (per 1000)','Arable (%)','Crops (%)','Other (%)','Climate','Birthrate','Deathrate','Agriculture','Industry','Service']

for i in cols:

    world[i] = world[i].str.replace(",",".")

    world[i] = world[i].astype('float64')
#Number of countries by region

region_counts = pd.DataFrame(world.Region.value_counts().reset_index()) 

region_counts = region_counts.rename(columns={"index":"Region","Region":"Number of countries"})

region_counts = region_counts.sort_values(by="Number of countries",ascending=False)

sns.barplot(data=region_counts,y="Region",x="Number of countries",palette="RdBu_r")

plt.title("Number of countries (Region-wise)")

plt.show()
#Boxplot of population of countries by region

sort_index_viz_2 = world.groupby("Region")["Population"].median().sort_values(ascending=False).index

viz_2 = sns.catplot(data=world,y="Region",x="Population",kind="box",color="#DECBE4",height=5,aspect=3,order=sort_index_viz_2)

viz_2.set(xscale="log")

plt.title("Population of countries by region")

plt.show()
#Total population % by region

pop_percent = world.groupby("Region")["Population"].sum().sort_values(ascending=False)/(world.Population.sum())

pop_percent = (round(pop_percent,2))*100

pop_percent
#Top 5 countries in terms of population

world[["Country","Region","Population"]].sort_values(by="Population",ascending=False).head(5).set_index("Country")
#Boxplot of area of countries by region

sort_index_viz_3 = world.groupby('Region')['Area (sq. mi.)'].median().sort_values(ascending=False).index

viz_3 = sns.catplot(data=world,y="Region",x="Area (sq. mi.)",kind="box",color="skyblue",height=5,aspect=3,order=sort_index_viz_3)

viz_3.set(xscale="log")

plt.title("Area of countries by region")

plt.xlabel("Area (Square Miles)")

plt.show()
#Top 5 countries in terms of area

world[["Country","Region","Area (sq. mi.)"]].sort_values(by="Area (sq. mi.)",ascending=False).head(5).set_index("Country")
#Boxplot of population densities of countries by region

sort_index_viz_4 = world.groupby('Region')['Pop. Density (per sq. mi.)'].median().sort_values(ascending=False).index

viz_4 = sns.catplot(data=world,y="Region",x="Pop. Density (per sq. mi.)",kind="box",color="#CCEBC5",height=5,aspect=3,order=sort_index_viz_4)

viz_4.set(xscale="log")

plt.title("Population densities of countries by region")

plt.xlabel("Population density (per Square Miles)")

plt.show()
#Top 5 countries in terms of population density

world[["Country","Region","Pop. Density (per sq. mi.)"]].sort_values(by="Pop. Density (per sq. mi.)",ascending=False).head(5).set_index("Country")
#Boxplot of Coastline (coast/area ratio) of countries by region

sort_index_viz_5 = world.groupby('Region')['Coastline (coast/area ratio)'].median().sort_values(ascending=False).index

viz_5 = sns.catplot(data=world,y="Region",x="Coastline (coast/area ratio)",kind="box",color="#FFFFCC",height=5,aspect=3,order=sort_index_viz_5)

viz_5.set(xscale="log")

plt.title("Coastline (coast/area) of countries by region")

plt.xlabel("Coastline (coast/area ratio)")

plt.show()
#Top 5 countries in terms of coastline

world[["Country","Region","Coastline (coast/area ratio)"]].sort_values(by='Coastline (coast/area ratio)',ascending=False).head(5).set_index("Country")
#Boxplot of Net migration of countries by region

sort_index_mig = world.groupby('Region')['Net migration'].median().sort_values(ascending=False).index

mig = sns.catplot(data=world,y="Region",x="Net migration",kind="box",color="green",height=5,aspect=3,order=sort_index_mig)

plt.title("Net migration rate of countries by region")

plt.xlabel("Net migration rate")

plt.show()
#Boxplot of Infant mortality rate of countries by region

sort_index_viz_6 = world.groupby('Region')['Infant mortality (per 1000 births)'].median().sort_values(ascending=False).index

viz_6 = sns.catplot(data=world,y="Region",x="Infant mortality (per 1000 births)",kind="box",color="#FDDAEC",height=5,aspect=3,order=sort_index_viz_6)

plt.title("Infant mortality rate of countries by region")

plt.xlabel("Infant mortality (per 1000 births)")

plt.show()
#Top 5 countries in terms of infant mortality rate

world[["Country","Region","Infant mortality (per 1000 births)"]].sort_values(by='Infant mortality (per 1000 births)',ascending=False).head(5).set_index("Country")
#Boxplot of GDP per capita of countries by region

sort_index_viz_7 = world.groupby('Region')['GDP ($ per capita)'].median().sort_values(ascending=False).index

viz_7 = sns.catplot(data=world,y="Region",x="GDP ($ per capita)",kind="box",color="#E5D8BD",height=5,aspect=3,order=sort_index_viz_7)

plt.title("GDP (per capita) of countries by region")

plt.xlabel("GDP ($ Per Capita)")

plt.show()
#Top 5 countries in terms of GDP per capita

world[["Country","Region","GDP ($ per capita)"]].sort_values(by='GDP ($ per capita)',ascending=False).head(5).set_index("Country")
#Boxplot of Literacy (%) of countries by region

sort_index_viz_8 = world.groupby('Region')['Literacy (%)'].median().sort_values(ascending=False).index

viz_8 = sns.catplot(data=world,y="Region",x="Literacy (%)",kind="box",color="#FBB4AE",height=5,aspect=3,order=sort_index_viz_8)

plt.title("Literacy (%) of countries by region")

plt.xlabel("Literacy (%)")

plt.show()
#Bottom 5 countries in terms of Literacy (%) - With non-nan values

lit_table = world[["Country","Region","Literacy (%)"]].sort_values(by='Literacy (%)',ascending=False).set_index("Country")

lit_table[~lit_table['Literacy (%)'].isna()].sort_values(by='Literacy (%)').head(5)
#Boxplot of Phones (per 1000) of countries by region

sort_index_viz_9 = world.groupby('Region')['Phones (per 1000)'].median().sort_values(ascending=False).index

viz_9 = sns.catplot(data=world,y="Region",x="Phones (per 1000)",kind="box",color="#FED9A6",height=5,aspect=3,order=sort_index_viz_9)

plt.title("Phones (per 1000) of countries by region")

plt.xlabel("Phones (per 1000)")

plt.show()
#Bottom 5 countries in terms of phones (per 1000) - With non-nan values

phone_table = world[["Country","Region","Phones (per 1000)"]].sort_values(by='Phones (per 1000)',ascending=False).set_index("Country")

phone_table[~phone_table['Phones (per 1000)'].isna()].sort_values(by='Phones (per 1000)').head(5)
#Boxplot of Arable land (%) countries by region

sort_index_viz_10 = world.groupby('Region')['Arable (%)'].median().sort_values(ascending=False).index

viz_10 = sns.catplot(data=world,y="Region",x="Arable (%)",kind="box",color="violet",height=5,aspect=3,order=sort_index_viz_10)

plt.title("Arable land (%) of countries by region")

plt.xlabel("Arable land (%)")

plt.show()
#Top 5 countries in terms of arable land %

world[["Country","Region","Arable (%)"]].sort_values(by='Arable (%)',ascending=False).head(5).set_index("Country")
#Boxplot of Birthrate of countries by region

sort_index_viz_11 = world.groupby('Region')['Birthrate'].median().sort_values(ascending=False).index

viz_11 = sns.catplot(data=world,y="Region",x="Birthrate",kind="box",color="grey",height=5,aspect=3,order=sort_index_viz_11)

plt.title("Birthrate of countries by region")

plt.xlabel("Birthrate")

plt.show()
#Top 5 countries in terms of birthrate

world[["Country","Region","Birthrate"]].sort_values(by='Birthrate',ascending=False).head(5).set_index("Country")
#Boxplot of Deathrate of countries by region

sort_index_viz_12 = world.groupby('Region')['Deathrate'].median().sort_values(ascending=False).index

viz_12 = sns.catplot(data=world,y="Region",x="Deathrate",kind="box",color="yellow",height=5,aspect=3,order=sort_index_viz_12)

plt.title("Deathrate of countries by region")

plt.xlabel("Deathrate")

plt.show()
#Top 5 countries in terms of deathrate

world[["Country","Region","Deathrate"]].sort_values(by='Deathrate',ascending=False).head(5).set_index("Country")
#Correlation matrix of numeric columns

corr_matrix = world.corr()

sns.heatmap(corr_matrix,cmap='PuOr')
#Correlogram with regression

first_cor = world[["GDP ($ per capita)","Literacy (%)","Phones (per 1000)","Birthrate","Deathrate","Infant mortality (per 1000 births)"]]

sns.pairplot(first_cor,kind="reg")
#Filter data for BRICS countries

brics = world.Country.isin(["Brazil","Russia","India","China","South Africa"])

brics = world[brics]

#Define function to make plots for BRICS countries

def brics_function(y,title):

    palette = {"Brazil":"#009C3B","Russia":"#0033A0","India":"#FF9933","China":"#DE2910","South Africa":"#000000"}

    sns.barplot(data=brics,x="Country",y=y,palette=palette,order=["Brazil","Russia","India","China","South Africa"])

    plt.ylabel("")

    plt.xlabel("")

    plt.title(title)

    plt.show()

#Generate multiple plots using for loop

brics_dict = {"Population":"#1 BRICS Population (In Billion)","Area (sq. mi.)":"#2 BRICS Area (In Square Miles)","Pop. Density (per sq. mi.)":"#3 BRICS Population Density (In Square Miles)","Coastline (coast/area ratio)":"#4 BRICS Coastline (Coast/Area ratio)","Net migration":"#5 BRICS Net Migration Rate",'Infant mortality (per 1000 births)':"#6 BRICS Infant Mortality (Per 1000 births)", 'GDP ($ per capita)':"#7 BRICS GDP ($ per capita)",'Literacy (%)':"#8 BRICS Literacy",'Phones (per 1000)':"#9 BRICS Number of Phones (Per 1000)",'Arable (%)':"#10 BRICS Arable Land %",'Crops (%)':"#11 BRICS Crops %",'Birthrate':"#12 BRICS Birthrate",'Deathrate':"#13 BRICS Deathrate"}

for key,value in brics_dict.items():

    brics_function(key,value)    
#Filter data for G7 countries

group_seven = world.Country.isin(["Canada","France","Germany","Italy","Japan","United Kingdom","United States"])

group_seven = world[group_seven]

#Define function to make plots for G7 countries

def group_seven_function(y,title):

    palette = {"Canada":"#FF0000","France":"#0055A4","Germany":"#FFCE00","Italy":"#008C45","Japan":"#BC002D","United Kingdom":"#00247D","United States":"#3C3B6E"}

    sns.barplot(data=group_seven,y="Country",x=y,palette=palette,order=["Canada","France","Germany","Italy","Japan","United Kingdom","United States"])

    plt.ylabel("")

    plt.xlabel("")

    plt.title(title)

    plt.show()

#Generate multiple plots using for loop for G7 countries

group_seven_dict = {"Population":"#1 G7 Population (x 100 Million)","Area (sq. mi.)":"#2 G7 Area (In Square Miles)","Pop. Density (per sq. mi.)":"#3 G7 Population Density (In Square Miles)","Coastline (coast/area ratio)":"#4 G7 Coastline (Coast/Area ratio)","Net migration":"#5 G7 Net Migration Rate",'Infant mortality (per 1000 births)':"#6 G7 Infant Mortality (Per 1000 births)", 'GDP ($ per capita)':"#7 G7 GDP ($ per capita)",'Literacy (%)':"#8 G7 Literacy",'Phones (per 1000)':"#9 G7 Number of Phones (Per 1000)",'Arable (%)':"#10 G7 Arable Land %",'Crops (%)':"#11 G7 Crops %",'Birthrate':"#12 G7 Birthrate",'Deathrate':"#13 G7 Deathrate"}

for key,value in group_seven_dict.items():

    group_seven_function(key,value)