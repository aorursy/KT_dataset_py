import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
suicides=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
suicides.head()
suicides.generation.unique() # For finding how many generations there are
suicides.columns
suicides.shape 
suicides.info()
suicides[' gdp_for_year ($) ']=suicides[' gdp_for_year ($) '].str.replace(',','').astype(int) 

# Because I can't convert the string with the commas in it, I need to remove them and then converting the string.

suicides.head()
suicides.isnull().any() # Verifying if there are any null values.
suicides.isnull().sum() # Counting all the null entries, adding 1 tot the sum everytime it's true that the value is null.
del suicides['HDI for year']
suicides.columns # It was successfully deleted.
len(suicides.country.unique()) 

# 101 countries.
suicides.rename(columns={'suicides/100k pop':'suicides/100k',' gdp_for_year ($) ':'gdp/year','gdp_per_capita ($)':'gdp/capita'},inplace=True)

# We already know that the currency is $.
suicides.drop('year',axis=1).describe()
suicides.groupby('country').suicides_no.sum().sort_values(ascending=False)

# The country with the most suicides is Russia, followed by USA and Japan

# The last 2 countries have 0 suicides, so we don't need them in our analysis about the number of suicides 
suicides=suicides[(suicides.country!='Saint Kitts and Nevis') & (suicides.country!='Dominica')]
total_suicides=pd.DataFrame(suicides.groupby('country').suicides_no.sum().sort_values(ascending=False))

# I'm gonna save the results to a DataFrame so I can plot them
total_suicides=total_suicides.reset_index()

# I don't want countries as index
total_suicides.head()
plt.figure(figsize=(10,20))

sns.barplot(y='country',x='suicides_no',data=total_suicides)

plt.title('Number of suicides aprox. 1985-2016') # It's aprox because the years for each country aren't quite the same

plt.ylabel('Countries')

plt.xlabel('Number of suicides')
plt.figure(figsize=(10,20))

sns.barplot(y=total_suicides[total_suicides.suicides_no<7000].country,

            x=total_suicides[total_suicides.suicides_no<7000].suicides_no,data=total_suicides)

plt.title('Number of suicides aprox. 1985-2016')

plt.ylabel('Countries')

plt.xlabel('Number of suicides')



# I chosed the suicides_no < 7000 just by looking at the figure and trying multiple times another values,

# finding this number appropiate.
from bokeh.io import output_notebook, output_file, show

from bokeh.plotting import figure

output_notebook()



suicides_globally=suicides.groupby('year').sum().suicides_no.values

years=suicides.year.sort_values(ascending=True).unique()

years



p1 = figure(plot_height=400, plot_width=900, title='Evolution of suicides over the world per year', tools='pan,box_zoom')



p1.line(x=years, y=suicides_globally, line_width=2, color='aquamarine')

p1.circle(x=years, y=suicides_globally, size=5, color='green')

show(p1)

    
len(suicides[suicides.year==2016].country.unique())

#There are only 16 countries studied this year, so we don't need the value in our analysis.
suicides_globally=suicides[suicides.year!=2016].groupby('year').sum().suicides_no.values

years=suicides[suicides.year!=2016].year.sort_values(ascending=True).unique()

p1 = figure(plot_height=400, plot_width=900, title='Evolution of suicides over the world per year', tools='pan,box_zoom')



p1.line(x=years, y=suicides_globally, line_width=2, color='aquamarine')

p1.circle(x=years, y=suicides_globally, size=5, color='green')

show(p1)
suicides_gender=pd.DataFrame(suicides.groupby(['country','sex']).suicides_no.sum())

suicides_gender=suicides_gender.reset_index()

suicides_gender.head()
plt.figure(figsize=(10,25))

sns.barplot(x=suicides_gender[suicides_gender.suicides_no > 7000].suicides_no,

            y=suicides_gender[suicides_gender.suicides_no > 7000].country,data=suicides_gender,hue='sex')



plt.figure(figsize=(10,25))

sns.barplot(x=suicides_gender[suicides_gender.suicides_no < 7000].suicides_no,

            y=suicides_gender[suicides_gender.suicides_no < 7000].country,data=suicides_gender,hue='sex')
suicides.groupby('sex').suicides_no.sum()
suicides.groupby('sex').suicides_no.sum().male/suicides.groupby('sex').suicides_no.sum().female * 100
genders=suicides.groupby(['year','sex']).sum().suicides_no

female=genders.loc[:2015,'female'].values

male=genders.loc[:2015,'male'].values





p2=figure(plot_height=500,plot_width=900,title='Evolution of number of suicides by gender globally',tools='pan, box_zoom')

p2.circle(x=years,y=male,color='purple',size=5)

p2.line(x=years,y=male,color='purple',legend='male',line_width=2)

p2.circle(x=years,y=female,color='orange',size=5)

p2.line(x=years,y=female,color='orange',legend='female',line_width=2)

show(p2)

suicides.groupby('age').sum().suicides_no
plt.figure(figsize=(10,5))

suicides_age=suicides.groupby('age').sum().suicides_no.values

age=suicides.groupby('age').sum().reset_index().age.values

sns.barplot(x=age,y=suicides_age)

plt.title('Number of suicides by age')
suicides_age=suicides.groupby(['year','age']).sum().suicides_no

suicides_age
age15_24=suicides_age.loc[:2015,'15-24 years'].values

age25_34=suicides_age.loc[:2015,'25-34 years'].values

age35_54=suicides_age.loc[:2015,'35-54 years'].values

age5_14=suicides_age.loc[:2015,'5-14 years'].values

age_above75=suicides_age.loc[:2015,'75+ years'].values
p3=figure(plot_height=500,plot_width=1000,title='Evolution of number of suicides by age globally',tools='pan,box_zoom')

p3.circle(x=years,y=age35_54,color='darkorchid',size=5)

p3.circle(x=years,y=age25_34,color='turquoise',size=5)

p3.circle(x=years,y=age15_24,color='limegreen',size=5)

p3.circle(x=years,y=age5_14,color='tomato',size=5)

p3.circle(x=years,y=age_above75,color='blue',size=5)

p3.line(x=years,y=age35_54,color='darkorchid',line_width=2,legend='35-54 years')

p3.line(x=years,y=age25_34,color='turquoise',line_width=2,legend='25-34 years')

p3.line(x=years,y=age15_24,color='limegreen',line_width=2,legend='15-24 years')

p3.line(x=years,y=age5_14,color='tomato',line_width=2,legend='5-14 years')

p3.line(x=years,y=age_above75,color='blue',line_width=2,legend='75+ years')

show(p3)
suicides.groupby('generation').sum().suicides_no
plt.figure(figsize=(10,5))

generations=suicides.groupby('generation').sum().suicides_no.reset_index().generation

suicides_gen=suicides.groupby('generation').sum().suicides_no.values

sns.barplot(x=generations,y=suicides_gen)

plt.xlabel('Generation')

plt.title('Number of suicides by generation')
plt.figure(figsize=(10,8))

sns.heatmap(suicides.corr(),annot=True,cmap="BuPu")
sns.pairplot(suicides.drop('year',axis=1),hue='sex',palette='bright')
gdp_year=suicides[suicides.year==2015]['gdp/year'].drop_duplicates().values

pop=suicides[suicides.year==2015].groupby('country').sum().population.values

no_suicides=suicides[suicides.year==2015].groupby('country').sum().suicides_no.values

ctry=suicides[suicides.year==2015].groupby('country').sum().reset_index().country.values



s=pd.DataFrame()

s['country']=ctry

s['pop']=pop

s['no_suicides']=no_suicides

s['gdp_year']=gdp_year



s.head()
from bokeh.models import HoverTool

from bokeh.models import ColumnDataSource

source=ColumnDataSource(s)

p4=figure(plot_height=500,plot_width=900,title='Identifying countries by population, number of suicides and GDP in 2015',tools='pan,box_zoom,wheel_zoom,reset')

p4.diamond(x='pop',y='no_suicides',size=10,color='green',source=source)

p4.add_tools(HoverTool(tooltips=[('population','@pop'),('number suicides','@no_suicides'),('country','@country'),('GDP','@gdp_year')]))

p4.xaxis.axis_label='Population'

p4.yaxis.axis_label='Number of suicides'

p4.xaxis.axis_label_standoff = 30 # Distance of the xaxis label from the xaxis 

show(p4)