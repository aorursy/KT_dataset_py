"""We are working on the data set of Suicidals in  Different Countries  based on the Age.

The Data Columns are : Shape is (27820, 12) and  total data size is 333840

    country : - Which country they belong to 

    year:- Which year the data has been taken.

    sex :- Determines the gender (IPS are male and female) 

    age:- Age is given in the form of groups (5-14 ,15-24,25-34,35-54,55-74,75+)

    suicides_no:- No of people commited sucides based on Age,

    population:- Total population in that year

    suicides/100k pop : - No of sucides commited as per 100k population

    country-year:- Combination of country and year

    HDI for year:- The Human Development Index (HDI) is a statistical tool used to measure a country's overall achievement 

                in its social and economic dimensions. The social and economic dimensions of a country are based on the health

                of people, their level of education attainment and their standard of living.

    gdp_for_year ($):- Growth of that country for that particular year 

    gdp_per_capita ($):- GDP per capita is a measure of a country's economic output that accounts for its number of people. 

                        It divides the country's gross domestic product by its total population. That makes it the best 

                        measurement of a country's standard of living

    generation :-  Types of people based on which year they were born .

                      Generation Z' - 5-14 years

                      Generation X', :- 15-24 years

                      Millenials', - 25-34 years

                      Silent', :-  35-54 years

                      'Boomers' :-  55-74  years

                      'G.I. Generation', - 75+ years . It is changing from country to country

  

   """ 

    
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
import pandas as pd

df=pd.read_csv("../input/master.csv")

df.head()
#df.country.count

#df[['country','age','generation']]

df[27808:]
pdf=df.groupby('country')[['suicides_no','country']].count()

#pdf.columns

pdf.sort_values('suicides_no',ascending=False)[['suicides_no']]

#var = pdf.max(axis=1)

#var

#print(pdf.where(var,inplace=True))

#pdf.groupby('country')['country'].values

gb=df.groupby(['country'])

#gb.get_group['India']

#print(gb.first())

gp=gb.get_group('Cuba')

x=gp.groupby('year')

#.sum()

#x_axis=x['year']

y_axis=x['suicides_no']

print(y_axis)

#x.columns()

#x.keys()

#gb.get_group()



for i,value in x:

    print(i)

    #print(value)