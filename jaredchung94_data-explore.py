import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
df = pd.read_csv("../input/world-happiness-report/2020.csv")
# Checking if data imported by looking at the head
df.head()
# Looking at the columns header
df.columns
# Getting number of data rows
nRow, nCol = df.shape
print("Num of row:", nRow)
print("Num of col:", nCol)
# Looking at the datatypes in each Column, and number of null values 
df.info()
# Checking for outliers 
sns.distplot(df["Ladder score"])
df_top20 = df[:20].sort_values('Ladder score', ascending = True)
px.bar(df_top20, x='Ladder score', y='Country name',
       orientation='h',title="Top 20 happiest countries")
df_last20 = df.sort_values('Ladder score', ascending = True)
df_last20 = df_last20[:20]
px.bar(df_last20, x='Ladder score', y='Country name',
       orientation='h',title="Last 20 least happiest countries")
#scatter plot between Logged GDP and Ladder score
var = 'Logged GDP per capita'
data = pd.concat([df["Ladder score"], df[var]],axis=1)
data.plot.scatter(x=var, y='Ladder score')
#scatter plot between Social support and Ladder score
var = 'Social support'
data = pd.concat([df["Ladder score"], df[var]],axis=1)
data.plot.scatter(x=var, y='Ladder score')
#scatter plot between Healthy life expectancy and Ladder score
var = 'Healthy life expectancy'
data = pd.concat([df["Ladder score"], df[var]],axis=1)
data.plot.scatter(x=var, y='Ladder score')
#scatter plot between Freedom to make life choices and Ladder score
var = 'Freedom to make life choices'
data = pd.concat([df["Ladder score"], df[var]],axis=1)
data.plot.scatter(x=var, y='Ladder score')
#scatter plot between Perceptions of corruption and Ladder score
var = 'Perceptions of corruption'
data = pd.concat([df["Ladder score"], df[var]],axis=1)
data.plot.scatter(x=var, y='Ladder score')
corrmap = df.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmap, square = True, linecolor = 'black',cmap="YlGnBu")
# Getting the chosen variables 
cols = ['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy'
       ,'Freedom to make life choices','Perceptions of corruption','Dystopia + residual']
sns.set()
sns.pairplot(df[cols],height = 3,kind = 'reg',corner = True)
plt.show()
data_2015 = pd.read_csv("../input/world-happiness-report/2015.csv")
data_2016 = pd.read_csv("../input/world-happiness-report/2016.csv")
data_2017 = pd.read_csv("../input/world-happiness-report/2017.csv")
data_2018 = pd.read_csv("../input/world-happiness-report/2018.csv")
data_2019 = pd.read_csv("../input/world-happiness-report/2019.csv")
data_2020 = pd.read_csv("../input/world-happiness-report/2020.csv")
data_full = [data_2015,data_2016,data_2017,data_2018,data_2019,data_2020]
for item in data_full:
    print(item.shape)
for item in data_full:
    print(item.columns)
data_2017 = data_2017.rename(columns={'Happiness.Rank':'Happiness Rank'})
data_2018 = data_2018.rename(columns={'Country or region': 'Country',
                            'Overall rank': 'Happiness Rank'})
data_2019 = data_2019.rename(columns={'Country or region': 'Country',
                            'Overall rank': 'Happiness Rank'})
data_2020 = data_2020.rename(columns={'Country name': 'Country'})

print("Success")
# Creating new columns for year 2020 dataset
data_2020['Happiness Rank'] = data_2020['Ladder score'].rank(ascending = False)
# Getting the selected columns
data_2015 = data_2015[['Country','Happiness Rank']]
data_2016 = data_2016[['Country','Happiness Rank']]
data_2017 = data_2017[['Country','Happiness Rank']]
data_2018 = data_2018[['Country','Happiness Rank']]
data_2019 = data_2019[['Country','Happiness Rank']]
data_2020 = data_2020[['Country','Happiness Rank']]
# Getting the unique number of country in each data set
data_full = [data_2015,data_2016,data_2017,data_2018,data_2019,data_2020]
for item in data_full:
    print(item.Country.nunique())
Merged = pd.merge(data_2015,data_2016, on=['Country'],how='inner')
Merged = Merged.rename(columns={'Happiness Rank_x':'2015','Happiness Rank_y':'2016' })
Merged = pd.merge(Merged,data_2017, on=['Country'],how='inner')
Merged = Merged.rename(columns={'Happiness Rank':'2017'})
Merged = pd.merge(Merged,data_2018, on=['Country'],how='inner')
Merged = Merged.rename(columns={'Happiness Rank':'2018'})
Merged = pd.merge(Merged,data_2019, on=['Country'],how='inner')
Merged = Merged.rename(columns={'Happiness Rank':'2019'})
Merged = pd.merge(Merged,data_2020, on=['Country'],how='inner')
Merged = Merged.rename(columns={'Happiness Rank':'2020'})
# Changing type for consistency 
Merged['2020'] = Merged['2020'].astype('int')
# Changing index values
Merged = Merged.set_index('Country')
# The top 10 country in the year 2015 and how it changes over the years
Merged[:10].T.plot()
# Looking for specific country ranking
Var = 'Finland'
Merged.loc[Var].T.plot()
