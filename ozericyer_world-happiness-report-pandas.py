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
data_15=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
data_16=pd.read_csv("/kaggle/input/world-happiness/2016.csv")
data_17=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
data_18=pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data_19=pd.read_csv("/kaggle/input/world-happiness/2019.csv")
data15=data_15.copy()
data16=data_16.copy()
data17=data_17.copy()
data18=data_18.copy()
data19=data_19.copy()  #We copy original data
data15.columns
data15.drop(['Happiness Rank','Standard Error','Economy (GDP per Capita)','Family','Dystopia Residual'],axis=1,inplace=True)
#we deleted unnecessary columns
data15.rename(columns={"Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Corruption"},inplace=True)
#We rename columns what we want.
data15['Year'] = '2015' #We add column Year 
data15 = data15[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']] 
#we have determined the columns order
data15  #final version of the report 2015.
data16.columns
data16.drop(['Happiness Rank','Lower Confidence Interval','Upper Confidence Interval','Economy (GDP per Capita)','Family','Dystopia Residual'],axis=1,inplace=True)
#we deleted unnecessary columns
data16.rename(columns={"Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Corruption"},inplace=True)
#We rename columns what we want.
data16['Year'] = '2016' #We add a column 'Year'.And order columns
data16 = data16[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']]
data16 #Final version of the report 2016
data17.columns
data17.drop(['Happiness.Rank','Whisker.high','Whisker.low','Economy..GDP.per.Capita.','Family','Dystopia.Residual',],axis=1,inplace=True)
#we delete unnecessary columns,
data17.rename(columns={"Happiness.Score":"Happiness Score","Health..Life.Expectancy.":"Health","Trust..Government.Corruption.":"Corruption"},inplace=True)
#We remane columns.
data17['Year'] = '2017' #We add a new column and order columns
data17 = data17[['Year','Country','Happiness Score','Health','Freedom','Corruption','Generosity']]
data17 #Final version of the report 2017
data18.columns
data18.rename(columns={"Country or region":"Country","Score":"Happiness Score","Healthy life expectancy":"Health","Freedom to make life choices":"Freedom","Perceptions of corruption":"Corruption"},inplace=True)
#We rename columns,
data18.drop(['Overall rank','GDP per capita','Social support'],axis=1,inplace=True) #We delete unnecessary columns.
data18['Year'] = '2018' #We add columns 'Year'.And order columns
data18 = data18[['Year','Country','Happiness Score','Health','Freedom','Corruption','Generosity']]
data18 ##Final version of the report 2018
data19.columns
data19.drop(['Overall rank','GDP per capita','Social support'],axis=1,inplace=True) #We delete unnecessary columns
data19.rename(columns={"Country or region":"Country","Score":"Happiness Score","Healthy life expectancy":"Health","Freedom to make life choices":"Freedom","Perceptions of corruption":"Corruption"},inplace=True)
#We rename columns what we want.
data19['Year'] = '2019' #Add a new columns and order all columns
data19 = data19[['Year','Country','Happiness Score','Health','Freedom','Corruption','Generosity']]
data19 #Final version of the report 2019
a=list(data16.Country.unique()) #we look at sorted countries.
sorted(a)
alldata=pd.concat([data15,data16,data17,data18,data19],axis=0,ignore_index=True)
alldata =alldata[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']]
alldata #we combined all reports into a single dataframe
# print(alldata[alldata.Country=='Trinidad & Tobago'])
# print(alldata[alldata.Country=='Taiwan Province of China'])
# print(alldata[alldata.Country=='Hong Kong S.A.R., China'])
# print(alldata[alldata.Country=='Northern Cyprus'])
# print(alldata[alldata.Country=='North Macedonia']) 
#Changing different names for the same countries for consistency.
alldata.loc[507,'Country'] = 'Trinidad and Tobago'
alldata.loc[664,'Country'] = 'Trinidad and Tobago'
alldata.loc[385,'Country'] = 'Hong Kong'
alldata.loc[527,'Country'] = 'North Cyprus'
alldata.loc[689,'Country'] = 'North Cyprus'
alldata.loc[709,'Country'] = 'Macedonia'
alldata.loc[347,'Country'] = 'Taiwan'
#Changing different names for the same countries for consistency.
alldata.columns

d = data15.iloc[:,1:3] 
alldata = pd.merge(alldata,d, on = 'Country', how = 'left')

#Take the country and region information.

alldata
alldata.drop(["Region_x"],axis=1,inplace=True)
alldata.rename(columns={"Region_y":"Region"},inplace=True) #We delete unnecessary columns.And order columns
alldata =alldata[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']]
alldata #Final version of alldata(2015,2016,2017,2018,2019)
alldata.isna().sum() #There are 18 NaN values of region and one value of corruption
alldata[alldata.Region.isna()].Country
#We find missing region informations
alldata.loc[233,'Region'] = 'Sub-Saharan Africa'
alldata.loc[407,'Region'] = 'Sub-Saharan Africa'
alldata.loc[567,'Region'] = 'Sub-Saharan Africa'
alldata.loc[737,'Region'] = 'Sub-Saharan Africa'
alldata.loc[270,'Region'] = 'Sub-Saharan Africa'
alldata.loc[425,'Region'] = 'Sub-Saharan Africa'
alldata.loc[588,'Region'] = 'Sub-Saharan Africa'
alldata.loc[738,'Region'] = 'Sub-Saharan Africa'
alldata.loc[300,'Region'] = 'Sub-Saharan Africa'
alldata.loc[461,'Region'] = 'Sub-Saharan Africa'
alldata.loc[623,'Region'] = 'Sub-Saharan Africa'
alldata.loc[781,'Region'] = 'Sub-Saharan Africa'
alldata.loc[254,'Region'] = 'Sub-Saharan Africa'
alldata.loc[745,'Region'] = 'Sub-Saharan Africa'
alldata.loc[364,'Region'] = 'Latin America and Caribbean'  
alldata.loc[518,'Region'] = 'Latin America and Caribbean'
alldata.loc[172,'Region'] = 'Latin America and Caribbean'
alldata.loc[209,'Region'] = 'Latin America and Caribbean'
alldata["Corruption"].mean()
alldata[alldata.Corruption.isna()]
alldata["Corruption"].fillna(alldata["Corruption"].mean(),inplace=True) #We write missing corruption as mean.
alldata.isna().sum() #We havent NaN values now.
alldata.info() #alldata information
alldata.describe().T #alldata describe.
alldata["Region"].value_counts() 
alldata.Region.unique()
len(alldata.Region.unique())
koralasyon=alldata.corr() #We find correlation of alldata.
koralasyon
koralasyon[(koralasyon.abs()>0.5)&(koralasyon.abs()<1)]  #health and feedom affect happiness most.
corr_years=alldata.groupby("Year").corr()
corr_years
corr_years[(corr_years.abs()>0.5)&(corr_years.abs()<1)] #Generally,every year health and freedom affect happiness score most.
alldata["Region"].value_counts()
happ_averages=alldata.groupby('Country')['Happiness Score'].mean()
happ_averages
happ_averages.sort_values(ascending=False).head(3)
happ_averages.sort_values(ascending=False).tail(3)
corrup_averages=alldata.groupby('Country')['Corruption'].mean()
corrup_averages
corrup_averages.sort_values(ascending=False).head()
corrup_averages.sort_values(ascending=False).tail()
freedom_averages=alldata.groupby('Region')['Freedom'].mean()
freedom_averages.sort_values(ascending=False).head(1)
freedom_averages.sort_values(ascending=False).tail(1)
health_average = alldata.groupby('Region')['Health'].mean()
health_average.sort_values(ascending=False).tail(1)
alldata.groupby("Region").aggregate({'Happiness Score':'mean','Freedom':'mean', 'Corruption':'mean'}).T
