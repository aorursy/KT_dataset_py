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
Data_15 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

Data_16 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')

Data_17 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')

Data_18 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')

Data_19 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
Data_15 #Fetching all general data of 2015
Data_15.info() #checking values and the types
Data_15.columns # To see the titles of our data set
Data_15['Year'] = '2015' #Adding "Year" to our data after Drop some unnecessary columns

Data_15 = Data_15 [['Year','Country', 'Region', 'Happiness Rank', 'Happiness Score',

       'Standard Error', 'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',

       'Generosity', 'Dystopia Residual' ]]

Data_15
Data_15.drop(['Happiness Rank','Standard Error','Economy (GDP per Capita)','Family','Dystopia Residual'], axis=1, inplace=True)

Data_15
Data_15.columns = ['Year','Country','Region','Happiness','Health','Freedom','Corruption','Generosity']

Data_15
Data_16
Data_16.columns # To see the titles of our data set
Data_16['Year'] = '2016'

Data_16 = Data_16 [['Year','Country', 'Region', 'Happiness Rank', 'Happiness Score',

       'Lower Confidence Interval', 'Upper Confidence Interval',

       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',

       'Freedom', 'Trust (Government Corruption)', 'Generosity',

       'Dystopia Residual']]

Data_16
Data_16.drop(['Happiness Rank','Lower Confidence Interval','Upper Confidence Interval','Economy (GDP per Capita)',

              'Family','Dystopia Residual'], axis=1, inplace=True)

Data_16
Data_16.columns = ['Year','Country','Region','Happiness','Health','Freedom','Corruption','Generosity']

Data_16
Data_17.columns # To see the titles of our data set
Data_17['Year'] = '2017'

Data_17 = Data_17 [['Year','Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high',

       'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',

       'Health..Life.Expectancy.', 'Freedom', 'Generosity',

       'Trust..Government.Corruption.', 'Dystopia.Residual']]

Data_17
Data_17.drop(['Happiness.Rank','Whisker.high','Whisker.low','Economy..GDP.per.Capita.',

              'Family','Dystopia.Residual'], axis=1, inplace=True)

Data_17
Data_17.columns = ['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']

Data_17
Data_18.columns # To see the titles of our data set
Data_18['Year'] = '2018'

Data_18 = Data_18 [['Year','Overall rank', 'Country or region', 'Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption']]

Data_18
Data_18.drop(['Overall rank','GDP per capita','Social support'

              ], axis=1, inplace=True)

Data_18
Data_18.columns = ['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']

Data_18
Data_19.columns
Data_19['Year'] = '2019'

Data_19 = Data_19 [['Year','Overall rank', 'Country or region', 'Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption']]

Data_19
Data_19.drop (['Overall rank','GDP per capita','Social support'],axis=1, inplace=True)

Data_19
Data_19.columns = ['Year','Country','Happiness','Health','Freedom','Corruption','Generosity']

Data_19
# To concat our datas of 5 years

All_Data = pd.concat([Data_15,Data_16,Data_17,Data_18,Data_19], ignore_index=True, sort=False)

All_Data

sorted(list(All_Data.Country.unique()))
All_Data.columns # Showing All general columns
data1 = Data_15.iloc[:,1:3] 

All_Data = pd.merge(All_Data,data1, on = 'Country', how = 'left') 
All_Data
#Drop "Region_x" and rename "Region_y to Region" to all data

All_Data.drop(["Region_x"],axis=1,inplace=True)

All_Data.rename(columns={"Region_y":"Region"},inplace=True) 

All_Data =All_Data[['Year','Country','Region','Happiness','Health','Freedom','Corruption','Generosity']]
All_Data
# finding Nan values in each topic

All_Data.isna().sum()


All_Data[All_Data.Region.isna()].Country
#Completing regions which were NaN values



All_Data.loc[689,'Region'] = 'Mediterranean Sea'

All_Data.loc[527,'Region'] = 'Mediterranean Sea'

All_Data.loc[709,'Region'] = 'North Macedonia'

All_Data.loc[209,'Region'] = 'Central America'

All_Data.loc[518,'Region'] = 'Central America'

All_Data.loc[364,'Region'] = 'Central America'

All_Data.loc[172,'Region'] = 'Northern-South America'

All_Data.loc[385,'Region'] = 'Eastern Pearl River Delta'

All_Data.loc[233,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[254,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[270,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[300,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[347,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[407,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[425,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[461,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[507,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[567,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[588,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[623,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[664,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[737,'Region'] = 'Sub-Saharan Africa'  

All_Data.loc[738,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[745,'Region'] = 'Sub-Saharan Africa'

All_Data.loc[781,'Region'] = 'Sub-Saharan Africa'
All_Data[All_Data.Region.isna()].Country
All_Data.isna().sum()
# Filling avarage of NaN values of Generosity

All_Data["Generosity"].fillna(All_Data["Generosity"].mean(),inplace=True)

All_Data.isna().sum()
All_Data.describe().T
All_Data.info()
#calculating sort of region

All_Data.groupby("Region").mean()



len(All_Data.groupby("Region"))-1
#finding variables which affect happiness most.

correlation=All_Data.corr() 

correlation
correlation[(correlation.abs()>0.5)&(correlation.abs()<2)]


happy_score = All_Data.groupby("Country")["Happiness"].mean()

happy_score
happy_score.sort_values(ascending=False).head(3) #First 3 Happiest countries.
happy_score.sort_values(ascending=False).tail(3) #First 3 unhappiest countries.
corruption_rank = All_Data.groupby("Country")["Corruption"].mean()

corruption_rank
corruption_rank.sort_values(ascending=False).head() #Rank of most corruptions by countries 
corruption_rank.sort_values(ascending=False).tail() #Rank of less corruptions by countries 
freedom_rank = All_Data.groupby("Region")["Freedom"].mean()

freedom_rank
freedom_rank.sort_values(ascending=False).head(1)  #top two regions of freedom 

freedom_rank.sort_values(ascending=True).head(1)   # worsest region of freedom
unhealthy = All_Data.groupby("Region")["Health"].mean()

unhealthy
unhealthy.sort_values(ascending=True).head(1) #Unhealthiest region of countries
All_Data.groupby("Region").aggregate({'Happiness':'mean','Freedom':'mean', 'Corruption':'mean'})
