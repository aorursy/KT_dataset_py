import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv")
dataset.info()
#object = string
dataset.shape
#43776 rows, 6 columns
dataset['country'].value_counts(dropna=True)
dataset['population'].mean(),dataset['suicides_no'].mean()
#mean avg population and suicide rate
k = dataset[(dataset['age']=='15-24 years')&(dataset['suicides_no']>6000)][['country','suicides_no','sex','year']].sort_values(by='year',ascending=True)
k.style.background_gradient(cmap='Wistia')
#finding out the country with the highest suicide rate of youngsters (15-24 y.o.)
k = dataset[(dataset['age']=='15-24 years')&(dataset['suicides_no']<100)&(dataset['year']==2016)][['country','suicides_no','sex','year']].sort_values(by='suicides_no',ascending=True)
k.style.background_gradient(cmap='Wistia')
#finding out countries with the lowest suicide rate (little-to-no) of youngsters (15-24 y.o.) in 2016
dataset[['country','suicides_no']].groupby(['country']).agg('sum').sort_values(by='suicides_no').head(15).style.background_gradient(cmap='GnBu')
#the least amount of suicide cases sorted by country (Montserrat, Iraq and San Marino are among the countries with very low s.r.)
dataset[['suicides_no','country']].groupby(['country']).agg('sum').sort_values(by='suicides_no',ascending=False).head(15).style.background_gradient(cmap='cividis')
#unfortunately, countries like Russia, USA and Japan have notoriously high suicide rates among population (overall stats)
dataset[['suicides_no','year']].groupby(['year']).agg('sum').head(15).sort_values(by='suicides_no',ascending=False).style.background_gradient(cmap = 'spring')
#the highest suicide cases sorted by max value
dataset[['suicides_no','sex']].groupby(['sex']).agg(['sum','mean'])#.sort_values(by='suicides_no',ascending=False)
import plotly.express as px
df = px.data.gapminder()
df.head(5)
#getting new df for plotly visualizations
dataset[['suicides_no','population']] = dataset[['suicides_no','population']].fillna(value=0)
#now we need to get rid of nan values by filling them with 0s 
f = pd.merge(dataset,df,on='country')
f = f[['country','suicides_no','year_x','population','iso_alpha']]
f.head(5)
#there it is important to merge datasets in order to not obtain only basic plotly data from gapminder
import plotly.express as px

plt.rcParams['figure.figsize'] = (20, 15)
plt.style.use('dark_background')

df = px.data.gapminder()
fig = px.choropleth(f,
                    locations="iso_alpha", 
                    color="suicides_no", 
                    hover_name="country",
                    animation_frame="year_x", 
                    range_color=[5,90],
                    scope='europe',
                    projection='robinson'
                    )
fig.show()
print(plt.style.available)
data_albania = dataset[dataset['country']=='Albania']
fig = px.bar(data_albania, x='year',y='suicides_no',color='age',height=500)
fig.show()

fig = px.bar(data_albania, x='sex',y='suicides_no',color='age',height=500)
fig.show()
#showing suicide stats in Albania sorted by age
#most of suicide cases occur at the age between 35 and 54
data_albania.sample(3)
data_kz = dataset[dataset['country']=='Kazakhstan']
fig = px.histogram(data_kz, x='year',y='suicides_no',color='sex',height=500,marginal='rug',hover_data=data_kz.columns)
fig.show()
#comparison of suicide rate sorted by gender in the Republic of Kazakhstan
#x axis - year, y axis - # of suicide cases
world_data = dataset[dataset['country']=='Albania']
world_data.head(3)
#new data frame for the area plot below
fig = px.area(world_data, x='year',y='suicides_no',color='sex',line_group='country',height=700)
fig.show()
#comparison of suicide rates by gender in Albania between 1985 and 2016
newdf1=dataset.groupby(['country','age'])['suicides_no'].mean()
newdf1 = pd.DataFrame(newdf1)

plt.rcParams['figure.figsize']=(15,30)

plt.subplot(3,1,3)
color = plt.cm.rainbow(np.linspace(0,1,6))
newdf1['suicides_no']['Kazakhstan'].plot.bar(color=color)
plt.title('Suicide Trend in Kazakhstan by age',fontsize=50)
plt.xticks(rotation=0)
newdf2=dataset.groupby(['country','age'])['suicides_no'].mean()
newdf2 = pd.DataFrame(newdf2)

plt.rcParams['figure.figsize']=(15,30)

plt.subplot(3,1,3)
color = plt.cm.spring(np.linspace(0,1,6))
newdf1['suicides_no']['Russian Federation'].plot.bar(color=color)
plt.title('Suicide Trend in Russia by age',fontsize=50)
plt.xticks(rotation=0)
newdf3=dataset.groupby(['country','age'])['suicides_no'].mean()
newdf3 = pd.DataFrame(newdf3)

plt.rcParams['figure.figsize']=(15,30)

plt.subplot(3,1,3)
color = plt.cm.spring(np.linspace(0,1,6))
newdf1['suicides_no']['San Marino'].plot.bar(color=color)
plt.title('Suicide Trend in San Marino by age',fontsize=50)
plt.xticks(rotation=0)
df4 = px.data.gapminder()
df4.head(5)
#new data frame for world map plot
dataset[['suicides_no','population']] = dataset[['suicides_no','population']].fillna(value=0)
#eliminating null values from the dataset
mw = pd.merge(dataset,df4,on='country')
mw = f[['country','suicides_no','year_x','population','iso_alpha']]
mw.head(5)
#merging datasets to avoid using only basic plotly data from gapminder
fig = px.scatter_geo(mw,locations='iso_alpha',color='suicides_no',hover_name='country',animation_frame='year_x',projection='natural earth',range_color=[5,90])
fig.show()
