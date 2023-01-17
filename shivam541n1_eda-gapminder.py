# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Importing the Data
df=pd.read_csv("/kaggle/input/gapminder/gapminder_tidy.csv")
df
# Unique values in countries
df.Country.unique()
# Total number of countries
df.Country.nunique()
df.Year.unique()
df.Year.nunique()
df.region.unique()
df.region.nunique()
# Checking for missing values
df.isna().sum()
df[df.gdp.isna()]
regions=df.region.unique()
regions
df_region=df.groupby(["region","Year"]).sum()
df_region
df_region.reset_index(inplace=True)
df_region
def region_population_share(year): 
    df_region=df.groupby(["region","Year"]).sum()
    df_region.reset_index(inplace=True)
    fig=px.pie(df_region[df_region.Year==year], values='population', names='region')
    fig.show()
# Regionwise Population share in 1964
region_population_share(1964)
# Regionwise Population share in 2013
region_population_share(2013)
df_change=df_region[df_region.Year.isin([df_region.Year.min(),df_region.Year.max()])]
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':100})
bar=sns.barplot(x="region", y="population",data=df_change,hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
# List of countries in all regions
for r in regions:
    print("\nList of Countreis in {} : \n".format(r))
    print(df[df.region==r].Country.unique())
for r in regions:
    df_r=df[df.region==r]
    df_2=df_r[df_r.Year==2013].sort_values(by="population",ascending=False)
    countries=df_2.head(10).Country
    df_10=df_r[df_r.Country.isin(countries)]
    fig = px.line(df_10, x='Year', y='population', color='Country',title="Top 10 countries with highest population in {}".format(r))
    fig.show()
for r in regions:
    df_r=df[df.region==r]
    fig = px.line(df_r, x='Year', y='gdp', color='Country',title=r)
    fig.show()
for r in regions:
    df_r=df[df.region==r]
    df_2=df_r[df_r.Year==2013].sort_values(by="gdp",ascending=False)
    countries=df_2.head(10).Country
    df_10=df_r[df_r.Country.isin(countries)]
    fig = px.line(df_10, x='Year', y='gdp', color='Country',title="Top 10 countries with highest gdp in {}".format(r))
    fig.show()
df.corr()
px.imshow(df.corr())
sns.pairplot(df)
df_region=df.groupby(["Year","region"]).median()
df_region.reset_index(inplace=True)
df_region
for r in regions:
    data=df_region[df_region.region==r]
    fig=px.scatter(data,x="gdp",y="life",title=r)
    fig.show()
for r in regions:
    data=df_region[df_region.region==r]
    fig=px.scatter(data,x="gdp",y="child_mortality",title=r)
    fig.show()
for r in regions:
    data=df_region[df_region.region==r]
    fig=px.scatter(data,x="gdp",y="fertility",title=r)
    fig.show()
def country_population_share(year,n=15): 
    # Arguments- year= Year ,n = Number of top countries to show in pie chart
    df_64=df[df.Year==year]
    df_64.sort_values(by="population",ascending=False,inplace=True)
    population=df_64.head(n).loc[:,["Country","population"]]
    others=df_64.iloc[(n+1):,:].population.sum()
    population=population.append({"Country":"others","population":others},ignore_index=True)
    fig=px.pie(population, values='population', names='Country')
    fig.show()
country_population_share(2011,20)
small_10 = df.groupby("Country").min().sort_values(by="gdp").head(10)
small_10
df_small=df[df.Country.isin(small_10.index)]
bar=sns.barplot(x="Country", y="gdp",data=df_small[df_small.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
bar=sns.barplot(x="Country", y="fertility",data=df_small[df_small.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
bar=sns.barplot(x="Country", y="child_mortality",data=df_small[df_small.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
bar=sns.barplot(x="Country", y="life",data=df_small[df_small.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
large_10 = df.groupby("Country").max().sort_values(by="gdp",ascending=False).head(10)
large_10
df_large=df[df.Country.isin(large_10.index)]
bar=sns.barplot(x="Country", y="gdp",data=df_large[df_large.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
bar=sns.barplot(x="Country", y="fertility",data=df_large[df_large.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
bar=sns.barplot(x="Country", y="child_mortality",data=df_large[df_large.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
bar=sns.barplot(x="Country", y="life",data=df_large[df_large.Year.isin([1964,2013])],hue="Year")
bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
def year_trend(c,n=1):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    data=df[df.Country==c]
    fig = make_subplots(rows=3, cols=2,subplot_titles=("Population", "Fertility", "Child Mortality", "GDP", "Life"))

    fig.add_trace(go.Scatter(x=data.Year, y=data.population,name="Population"),row=1, col=1)
    fig.add_trace(go.Scatter(x=data.Year, y=data.fertility,name="Fertility"),row=1, col=2)
    fig.add_trace(go.Scatter(x=data.Year, y=data.child_mortality,name="Child Mortality"),row=2, col=1)
    fig.add_trace(go.Scatter(x=data.Year, y=data.gdp,name="GDP"),row=2, col=2)
    fig.add_trace(go.Scatter(x=data.Year, y=data.life,name="Life"),row=3, col=1)

    fig.update_layout(height=1000, width=1000,title_text="{0} Yearly Trends for {1} ".format(n,c))
    fig.show()
year_trend("India")

n=1
for c in small_10.index: 
    year_trend(c,n)
    n+=1
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
n=1
for c in large_10.index: 
    year_trend(c,n)
    n+=1
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
