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
wh_2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

wh_2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

wh_2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

wh_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

wh_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
wh_2015.columns

wh_2015.drop(["Standard Error","Economy (GDP per Capita)","Family",'Dystopia Residual'],axis = 1,inplace = True)



wh_2015.columns
wh_2016.drop(["Economy (GDP per Capita)","Lower Confidence Interval","Family",'Upper Confidence Interval','Dystopia Residual'],axis = 1,inplace = True)
wh_2016.columns
wh_2015.head(10)

co_reg = pd.DataFrame([wh_2015['Country'],wh_2015['Region']])

country_region = co_reg.T

wh_2015.drop(['Happiness Rank','Region'],axis = 1,inplace = True)
wh_2016.drop(['Happiness Rank','Region'],axis = 1,inplace = True)
wh_2016.columns

wh_2015.columns

wh_1 = wh_2015.rename(columns = {'Country':'Country', 'Happiness Score':'2015_Happiness Score', 'Health (Life Expectancy)':'2015_Health (Life Expectancy)',

       'Freedom':'2015_Freedom', 'Trust (Government Corruption)':'2015_Trust (Government Corruption)', 'Generosity':'2015_Generosity'})
wh_2 = wh_2016.rename(columns = {'Country':'Country','Happiness Score':'2016_Happiness Score', 'Health (Life Expectancy)':'2016_Health (Life Expectancy)',

       'Freedom':'2016_Freedom', 'Trust (Government Corruption)':'2016_Trust (Government Corruption)', 'Generosity':'2016_Generosity'})

wh_2.head()

wh_1.head()
wh_2017.drop(['Happiness.Rank',"Whisker.high","Whisker.low","Economy..GDP.per.Capita.","Family","Dystopia.Residual"],axis =1,inplace = True)
wh_2017.columns = ['Country','Happiness Score', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity']

wh_2017.head()
# new_wh_2017 = wh_2017[['Country','Happiness Score', 'Health (Life Expectancy)', 'Freedom','Generosity','Trust (Government Corruption)']]

wh_3 = wh_2017.rename(columns = {'Country':'Country', 'Happiness Score':'2017_Happiness Score', 'Health (Life Expectancy)':'2017_Health (Life Expectancy)',

       'Freedom':'2017_Freedom', 'Trust (Government Corruption)':'2017_Trust (Government Corruption)', 'Generosity':'2017_Generosity'})

wh_3.head()
wh_2018.drop(['Overall rank',"GDP per capita",'Social support'],axis = 1,inplace=True)

wh_2018.columns = ['Country','Happiness Score', 'Health (Life Expectancy)', 'Freedom','Generosity','Trust (Government Corruption)']

wh_2018 = wh_2018[['Country','Happiness Score', 'Health (Life Expectancy)', 'Freedom','Trust (Government Corruption)','Generosity']]
wh_4 = wh_2018.rename(columns = {'Country':'Country', 'Happiness Score':'2018_Happiness Score', 'Health (Life Expectancy)':'2018_Health (Life Expectancy)',

       'Freedom':'2018_Freedom', 'Trust (Government Corruption)':'2018_Trust (Government Corruption)', 'Generosity':'2018_Generosity'})
wh_2019.drop(['Overall rank','GDP per capita','Social support'],axis = 1, inplace = True)
wh_2019.columns = ['Country','Happiness Score', 'Health (Life Expectancy)', 'Freedom','Generosity','Trust (Government Corruption)']

wh_2019 = wh_2019[['Country','Happiness Score', 'Health (Life Expectancy)', 'Freedom','Trust (Government Corruption)','Generosity']]
wh_5 = wh_2019.rename(columns = {'Country':'Country','Happiness Score':'2019_Happiness Score', 'Health (Life Expectancy)':'2019_Health (Life Expectancy)',

       'Freedom':'2019_Freedom', 'Trust (Government Corruption)':'2019_Trust (Government Corruption)', 'Generosity':'2019_Generosity'})

wh_1.info()

wh_2.info()

wh_3.info()

wh_4.info()

wh_5.info()
all = wh_1.merge(wh_2,on='Country',how = "left")

all = all.merge(wh_3,on='Country',how = "left")

all = all.merge(wh_4,on='Country',how = "left")

all = all.merge(wh_5,on='Country',how = "left")

all = all.merge(country_region,on='Country',how = "left")



all = all[['Country','Region', '2015_Happiness Score','2016_Happiness Score','2017_Happiness Score','2018_Happiness Score','2019_Happiness Score',

           '2015_Health (Life Expectancy)','2016_Health (Life Expectancy)', '2017_Health (Life Expectancy)','2018_Health (Life Expectancy)','2019_Health (Life Expectancy)', 

       '2015_Freedom','2016_Freedom','2017_Freedom','2018_Freedom','2019_Freedom',

           '2015_Trust (Government Corruption)','2016_Trust (Government Corruption)', '2017_Trust (Government Corruption)','2018_Trust (Government Corruption)','2019_Trust (Government Corruption)',

           '2015_Generosity','2016_Generosity','2017_Generosity','2018_Generosity','2019_Generosity']]

all.head()
all.isna().sum()
for i in all.columns[2:]:

    all[i].fillna(all[i].mean(),inplace = True)        

all.isna().sum()  
all.info

all.T
len(all.Region.groupby(all['Region']).count())
all.corr()
all.groupby('Region')['Country'].nunique()
all["5year_Happines"] = (all['2015_Happiness Score']+all['2016_Happiness Score']+all['2017_Happiness Score']+

                         all['2018_Happiness Score']+all['2019_Happiness Score'])/5

# print(all["5year_Happines"])

happiest = all.sort_values(by = "5year_Happines",ascending = False)

happiest.head(3)

happiest.tail(3)
all["5year_Corruption"] = (all['2015_Trust (Government Corruption)']+all['2016_Trust (Government Corruption)']+all['2017_Trust (Government Corruption)']+

                         all['2018_Trust (Government Corruption)']+all['2019_Trust (Government Corruption)'])/5
corruption = all.sort_values(by = '5year_Corruption',ascending = False)

corruption.head(1) #Singapore

corruption.tail(1) #Lithuania
all['Mean_Freedom'] = (all['2015_Freedom']+ all['2016_Freedom']+all ['2017_Freedom']+all['2018_Freedom']+all['2019_Freedom'])/5
freedom_region = all.groupby("Region")['Mean_Freedom'].mean()

freedom_region
for i in (freedom_region):

    if i == freedom_region.min():

        print(freedom_region[freedom_region == i])

    elif i == freedom_region.max():

        print(freedom_region[freedom_region == i])
all['Health'] = (all['2015_Health (Life Expectancy)']+ all['2016_Health (Life Expectancy)']+all ['2017_Health (Life Expectancy)']+all['2018_Health (Life Expectancy)']+all['2019_Health (Life Expectancy)'])/5
health_region = all.groupby("Region")['Health'].mean()

for i in (health_region):

    if i == health_region.min():

        print(health_region[health_region == i])
all.head(1)
all.groupby('Region')["5year_Happines"].mean()
all.groupby('Region')["Mean_Freedom"].mean()
all.groupby('Region')["5year_Corruption"].mean()
all.groupby('Region')["Health"].mean()