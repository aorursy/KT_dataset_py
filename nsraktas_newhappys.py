# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_2015 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')

data_2016 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data_2017 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

data_2018 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')

data_2019 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')
data_2015['Year'] = 2015

data_2016['Year'] = 2016

data_2017['Year'] = 2017

data_2018['Year'] = 2018

data_2019['Year'] = 2019
data_2015.info()
data_2016.info()
data_2017.info()
data_2018.info()
data_2019.info()
# We created a dictionary based on 2017-2018 for non-Country and Region data.



Country_Region_dich = {}



for i in data_2017.index:

    Country_Region_dich[data_2017.loc[i,'Country']] = data_2017.loc[i,'Region']





for i in data_2018.index:

    Country_Region_dich[data_2018.loc[i,'Country']] = data_2018.loc[i,'Region']
data_2015.columns
# The names of the columns have been changed



data_2015.rename(columns={"Country": "Country", 

                          'Happiness.Rank':'Happiness.Rank',

                          'Happiness.Score': "Happiness.score",

                          'Health..Life.Expectancy.': "Health" ,

                          "Freedom" : "Freedom",

                          'Trust..Government.Corruption.':"Corruption",

                          "Generosity" : "Generosity",

                          }, inplace=True)

# unwanted columns names deleted

data_2015 = data_2015.drop(columns = ['Whisker.high','Whisker.low','Economy..GDP.per.Capita.','Family','Dystopia.Residual'])

# Country check from Country Region_dict



for i in data_2015['Country']:    

    if i not in Country_Region_dich.keys():

        print(i)
# incorrect data changed



data_2015.Country.replace('Taiwan Province of China','Taiwan',inplace=True)

data_2015.Country.replace('Hong Kong S.A.R., China','Hong Kong',inplace=True)

data_2015['Region'] = np.nan    #  Region columns added
# Region information for 2015 found appropriate Region data from Country Region_dict



for i in data_2015.Country:

    data_2015.loc[    data_2015.Country == i  ,   'Region' ]     = Country_Region_dich[i]
# data_2016
data_2016.columns
# The names of the columns have been changed

data_2016.rename(columns={"Country or region": "Country",

                          "Region": "Region",

                          'Overall rank':'Happiness.Rank',

                          "Score": "Happiness.score",

                          "Healthy life expectancy": "Health",

                          "Freedom to make life choices" : "Freedom",

                          "Perceptions of corruption":"Corruption",

                          "Generosity" : "Generosity",

                          }, inplace=True)
#unwanted columns names deleted

data_2016 = data_2016.drop(columns = ['GDP per capita','Social support',])
# Country check from Country Region_dict



for i in data_2016['Country']:    # Country kontrolu yaptik

    if i not in Country_Region_dich.keys():

        print(i)
# incorrect data changed

Country_Region_dich['Trinidad & Tobago'] = 'Latin America and Caribbean'

Country_Region_dich['Northern Cyprus'] = 'Middle East and Northern Africa'

Country_Region_dich['North Macedonia'] = 'Central and Eastern Europe'

Country_Region_dich['Gambia'] = 'Sub-Saharan Africa'
data_2016['Region'] = np.nan
# data_2016.Country.replace('Taiwan Province of China','Taiwan',inplace=True)

# data_2016.Country.replace('Hong Kong S.A.R., China','Hong Kong',inplace=True)

# Region information for 2015 found appropriate Region data from Country Region_dict



for i in data_2016.Country:

    data_2016.loc[    data_2016.Country == i  ,   'Region' ]     = Country_Region_dich[i]
# data_2017
data_2017.columns
# The names of the columns have been changed

data_2017.rename(columns={"Country": "Country", 

                          "Region": "Region",

                          'Happiness Rank':'Happiness.Rank',

                          "Happiness Score": "Happiness.score",

                          "Health (Life Expectancy)": "Health",

                          "Freedom" : "Freedom",

                          "Trust (Government Corruption)":"Corruption",

                          "Generosity" : "Generosity",

                          }, inplace=True)

#unwanted columns names deleted

data_2017 = data_2017.drop(columns = ['Standard Error','Economy (GDP per Capita)','Family','Dystopia Residual'])
# Country check from Country Region_dict



for i in data_2017['Country']:    

    if i not in Country_Region_dich.keys():

        print(i)
# data_2018
data_2018.columns
# The names of the columns have been changed

data_2018.rename(columns={"Country": "Country",

                          "Region": "Region",

                          'Happiness Rank':'Happiness.Rank',

                          "Happiness Score": "Happiness.score",

                          "Health (Life Expectancy)": "Health",

                          "Freedom" : "Freedom",

                          "Trust (Government Corruption)":"Corruption",

                          "Generosity" : "Generosity",

                          }, inplace=True)
#unwanted columns names deleted

data_2018 = data_2018.drop(columns = ['Lower Confidence Interval','Upper Confidence Interval','Economy (GDP per Capita)','Family','Dystopia Residual'])
# Country check from Country Region_dict



for i in data_2018['Country']:    

    if i not in Country_Region_dich.keys():

        print(i)
# data_2019
data_2019.columns
# The names of the columns have been changed

data_2019.rename(columns={"Country or region": "Country",

                          "Region": "Region",

                          'Overall rank':'Happiness.Rank',

                          "Score": "Happiness.score",

                          "Healthy life expectancy": "Health",

                          "Freedom to make life choices" : "Freedom",

                          "Perceptions of corruption":"Corruption",

                          "Generosity" : "Generosity",

                          }, inplace=True)
#unwanted columns names deleted

data_2019 = data_2019.drop(columns = ['GDP per capita','Social support',])
data_2019['Region'] = np.nan
# Country check from Country Region_dict



for i in data_2019['Country']:    # Country kontrolu yaptik

    if i not in Country_Region_dich.keys():

        print(i)
# Region information for 2015 found appropriate Region data from Country Region_dict



for i in data_2019.Country:

    data_2019.loc[    data_2019.Country == i  ,   'Region' ]     = Country_Region_dich[i]
# data_2019
data_2015.columns
data_2015 = data_2015[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',

       'Freedom', 'Corruption', 'Generosity', 'Year']]
data_2016 = data_2016[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',

       'Freedom', 'Corruption', 'Generosity', 'Year']]
data_2017 = data_2017[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',

       'Freedom', 'Corruption', 'Generosity', 'Year']]
data_2018 = data_2018[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',

       'Freedom', 'Corruption', 'Generosity', 'Year']]
data_2019 = data_2019[['Country', 'Region', 'Happiness.Rank', 'Happiness.score', 'Health',

       'Freedom', 'Corruption', 'Generosity', 'Year']]
# data_2015.columns == data_2019.columns
data = pd.concat([data_2015,data_2016,data_2017,data_2018,data_2019], ignore_index=True)
data.info()
data.describe().T
data.isnull().sum()
data = data.fillna(data.mean())   # eksik verilere bulundugu sutunun ort atadik
data.isnull().sum()
set(data.Region)
len(set(data.Region))
data.corr()
data.groupby('Region')['Country'].count()
data.groupby("Country")["Happiness.score"].mean().sort_values(ascending = False).head(3)
data.groupby("Country")["Happiness.score"].mean().sort_values(ascending = False).tail(3)
data.groupby("Country")["Corruption"].mean().sort_values()
data.groupby("Country")["Corruption"].mean().sort_values().tail(1)
data.groupby("Country")["Corruption"].mean().sort_values().head(1)
data.groupby("Region")["Freedom"].mean().sort_values(ascending = False) 
data.groupby("Region")["Freedom"].mean().sort_values(ascending = False).head(1) 
data.groupby("Region")["Freedom"].mean().sort_values(ascending = False).tail(1) 
data.groupby("Region")["Health"].mean().sort_values().head(1)
data.groupby('Region')['Happiness.score','Freedom','Corruption'].mean()