import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
data = pd.read_csv('/kaggle/input/gapminder/gapminder_tidy.csv')
data
len(data.Year.value_counts())
data.info()
data.describe()
data.Country.value_counts()
observed_features = ['fertility','life','population','child_mortality','gdp']
plt.figure(figsize=(15, 15))

for i, col in enumerate(observed_features,1):
    plt.subplot(5,4, i)
    plt.hist(x=col,data=data)
    plt.xlabel(col)
plt.tight_layout()
for i in observed_features:
    data[i].fillna(data[i].mean(),inplace=True)
data.info()
plt.figure(figsize=(15, 15))

for i, col in enumerate(observed_features,1):
    plt.subplot(5,4, i)
    plt.boxplot(x=col,data=data)
    plt.xlabel(col)
plt.tight_layout()
data
for j in data.region.unique():
    new = data[data.region == j]
    
    for col in observed_features:
        fig = px.pie(new, values=col, names='Country',title=f"pie plot of {j}'s {col} for it's countries")
        fig.show()
fertility = pd.DataFrame(data.groupby('region')['fertility'].mean())
fig = px.bar(fertility, x=fertility.index, y="fertility", orientation='v',title='Region Wise Fertility',width=500,height=400)
fig.show()
life = pd.DataFrame(data.groupby('region')['life'].mean())
fig = px.bar(life, x=life.index, y="life", orientation='v',title='Region Wise Life Expectancy',width=500,height=400)
fig.show()
population = pd.DataFrame(data.groupby('region')['population'].mean())
fig = px.bar(population, x=population.index, y="population", orientation='v',title='Region Wise Population',width=500,height=400)
fig.show()
mortality = pd.DataFrame(data.groupby('region')['child_mortality'].mean())
fig = px.bar(mortality, x=mortality.index, y="child_mortality", orientation='v',title='Region Wise Child Mortality',width=500,height=400)
fig.show()
gdp = pd.DataFrame(data.groupby('region')['gdp'].mean())
fig = px.bar(gdp, x=gdp.index, y="gdp", orientation='v',title='Region Wise GDP',width=500,height=400)
fig.show()
sns.pairplot(data[observed_features])
sns.heatmap(data.corr())
fig = px.bar(data, x='Year', y="fertility", orientation='v',title='Trend of fertility of all regions',width=700,height=1500,facet_row='region')
fig.show()
fig = px.bar(data, x='Year', y="life", orientation='v',title='Trend of life expectancy of all regions',width=700,height=1500,facet_row='region')
fig.show()
fig = px.bar(data, x='Year', y="population", orientation='v',title='Trend of population of all regions',width=700,height=1500,facet_row='region')
fig.show()
fig = px.bar(data, x='Year', y="child_mortality", orientation='v',title='Trend of child mortality of all regions',width=700,height=1500,facet_row='region')
fig.show()
fig = px.bar(data, x='Year', y="gdp", orientation='v',title='Trend of gdp of all regions',width=700,height=1500,facet_row='region')
fig.show()
country = pd.DataFrame(data.groupby(['Country'])['fertility','life','population','child_mortality','gdp'].mean())
country
def max_min(col):
    maxi = country.loc[country[col] == country[col].max(),col]
    mini = country.loc[country[col] == country[col].min(),col]
    print(maxi,'\n',mini,'\n')

for i in country.columns:
    max_min(i)
import statsmodels.api as sm 
from statsmodels.formula.api import ols
lm = ols('fertility ~ Country',data = data).fit()
table = sm.stats.anova_lm(lm)
print(table)
lm = ols('life ~ Country',data = data).fit()
table = sm.stats.anova_lm(lm)
print(table)
lm = ols('population ~ Country',data = data).fit()
table = sm.stats.anova_lm(lm)
print(table)
lm = ols('child_mortality ~ Country',data = data).fit()
table = sm.stats.anova_lm(lm)
print(table)
lm = ols('gdp ~ Country',data = data).fit()
table = sm.stats.anova_lm(lm)
print(table)
