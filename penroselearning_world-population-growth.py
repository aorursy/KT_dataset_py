import pandas as pd

from population import live_data,growth_rate,population_density

import math

from datetime import datetime
df1 = pd.read_csv('../input/countries/countries.csv')
df1.head()
df2 = pd.read_html(live_data().text)

df2 = df2[0]
df2.head()
df = pd.merge(df1,df2,how="inner",on="Country")
df.info()
df.head()
df.drop("Area",axis=1,inplace=True)
df.head()
columns = ['2020 Population','2019 Population','Area (kmÂ²)']



for col in columns:

    df[col] = df[col].apply(lambda x: round(x/1000000,2))
df.head()
world_population = df["2020 Population"].sum()

f'{world_population/1000} Billion is the current world population'
df["Population Share %"] = df["2020 Population"].apply(lambda x: round((x/world_population)*100,1))
df.head()
df.rename(columns={'Rank':"Rank (Population)"},inplace=True)
df["Rank (Area)"] = df['Area (kmÂ²)'].rank(method='max',ascending=False)
df["Density (kmÂ²)"] = df.apply(lambda x: population_density(x["2020 Population"],x["Area (kmÂ²)"]),axis=1)
df["Rank (Density)"] = df['Density (kmÂ²)'].rank(method='max',ascending=False)
df.drop("2018 Density",axis=1,inplace=True)
df.head()
df["Growth Rate"] = df.apply(lambda x: growth_rate(x["2019 Population"],x["2020 Population"]),axis=1)
declining_population = df[df["Growth Rate"] < 0].sort_values("Growth Rate")

declining_population[["Country","Growth Rate"]].head(10)
continent = declining_population.groupby(by="Continent").size().sort_values(ascending=False)
f'''{continent.index[0]}' is the leading continent with most countries ({continent[0]}) having a declining population'''
dense_population = df[df["Rank (Density)"] <= 10].sort_values("Rank (Density)")

dense_population[["Country","Continent","Density (kmÂ²)"]]
world_population_2019 = df["2019 Population"].sum()

f'{world_population_2019/1000} Billion was the total world population in the Previous Year'
population_growth_rate = (1 - (world_population_2019/world_population)) * 100

f'{round(population_growth_rate,2)}% is the current population growth rate'
population_estimate_2050 = round(world_population * (math.exp ((population_growth_rate/100) * 30)))

f'{population_estimate_2050/1000} Billion is the Estimated Population by 2050'
continent = df.groupby(by='Continent')['2020 Population', '2019 Population', 'Area (kmÂ²)'].sum().reset_index()

continent["Growth Rate"] = continent.apply(lambda x: growth_rate(x["2019 Population"],x["2020 Population"]),axis=1)

continent["Density (kmÂ²)"] = continent.apply(lambda x: population_density(x["2020 Population"],x["Area (kmÂ²)"]),axis=1)

continent["Rank (Population)"] = continent['2020 Population'].rank(method='max',ascending=False)

continent["Rank (Area)"] = continent['Area (kmÂ²)'].rank(method='max',ascending=False)

continent["Rank (Density)"] = continent['Density (kmÂ²)'].rank(method='max',ascending=False)

continent.head()
today = datetime.now().date() 



with pd.ExcelWriter(f'world_population_{today}.xlsx') as writer:  

    df.to_excel(writer, sheet_name='World_Population')

    declining_population.to_excel(writer, sheet_name='Declining Population')

    dense_population.to_excel(writer, sheet_name='Top 10 - Population Density')

    continent.to_excel(writer, sheet_name='Data by Continent')