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
happiness_2019=pd.read_csv('/kaggle/input/world-happiness/2019.csv')
happiness_2018=pd.read_csv('/kaggle/input/world-happiness/2018.csv')
happiness_2017=pd.read_csv('/kaggle/input/world-happiness/2017.csv')
happiness_2016=pd.read_csv('/kaggle/input/world-happiness/2016.csv')
happiness_2015=pd.read_csv('/kaggle/input/world-happiness/2015.csv')
happiness_2019
new_happ_2019=happiness_2019[['Country or region','Score','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']]
new_happ_2019
new_happ_2019.rename(columns={'Country or region':'Country','Score':'Score_19','Healthy life expectancy':'Health_19','Freedom to make life choices':'Freedom_19','Generosity':'Generosity_19','Perceptions of corruption':'Corruption_19'}, inplace=True)
new_happ_2019
happiness_2018
new_happ_2018=happiness_2018[['Country or region','Score','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']]
new_happ_2018
new_happ_2018.rename(columns={'Country or region':'Country','Score':'Score_18','Healthy life expectancy':'Health_18','Freedom to make life choices':'Freedom_18','Generosity':'Generosty_18','Perceptions of corruption':'Corruption_18'}, inplace=True)
new_happ_2018
happiness_2017
new_happ_2017=happiness_2017[['Country','Happiness.Score','Health..Life.Expectancy.','Freedom','Generosity','Trust..Government.Corruption.']]
new_happ_2017
new_happ_2017.rename(columns={'Happiness.Score':'Score_17','Health..Life.Expectancy.':'Health_17','Freedom':'Freedom_17','Generosity':'Generosty_17','Trust..Government.Corruption.':'Corruption_17'} ,inplace=True)
new_happ_2017
happiness_2016
new_happ_2016=happiness_2016[['Country','Region','Happiness Score','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)']]
new_happ_2016
new_happ_2016.rename(columns={'Happiness Score':'Score_16','Health (Life Expectancy)':'Health_16','Freedom':'Freedom_16','Generosity':'Generosty_16','Trust (Government Corruption)':'Corruption_16'}, inplace=True)
new_happ_2016
happiness_2015
new_happ_2015=happiness_2015[['Country','Region','Happiness Score','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)']]
new_happ_2015
new_happ_2015.rename(columns={'Happiness Score':'Score_15','Health (Life Expectancy)':'Health_15','Freedom':'Freedom_15','Generosity':'Generosty_15','Trust (Government Corruption)':'Corruption_15'}, inplace=True)
new_happ_2015
new_happ_2019,new_happ_2018,new_happ_2017,new_happ_2016,new_happ_2015
data=pd.concat([new_happ_2015['Country'],new_happ_2016['Country'],new_happ_2017['Country'],new_happ_2018['Country'],new_happ_2019['Country']], ignore_index=True)
data
country=pd.DataFrame(data.unique())
country.rename(columns={0:'Country'}, inplace=True)
country
merged_data=pd.merge(country,new_happ_2015, how='left', on="Country")
merged_data
merged_data[merged_data['Region'].isna()]
merged_data.iloc[158,1]='Latin America and Caribbean'
merged_data.iloc[159,1]='Latin America and Caribbean'
merged_data.iloc[160,1]='Sub-Saharan Africa'
merged_data.iloc[161,1]='Sub-Saharan Africa'
merged_data.iloc[162,1]='Sub-Saharan Africa'
merged_data.iloc[163,1]='Sub-Saharan Africa'
merged_data.iloc[164,1]='Eastern Asia'
merged_data.iloc[165,1]='Eastern Asia'
merged_data.iloc[166,1]='Latin America and Caribbean'
merged_data.iloc[167,1]='Middle East and Northern Africa'
merged_data.iloc[168,1]='Central and Eastern Europe'
merged_data.iloc[169,1]='Sub-Saharan Africa'
merged_data
merged_data[merged_data['Region'].isna()]
merged_data
merged_data=pd.merge(merged_data,new_happ_2016, how='left', on="Country")
merged_data
merged_data=pd.merge(merged_data,new_happ_2017, how='left', on="Country")
merged_data
merged_data=pd.merge(merged_data,new_happ_2018, how='left', on="Country")
merged_data
merged_data=pd.merge(merged_data,new_happ_2019, how='left', on="Country")
merged_data
merged_data.drop('Region_y', axis=1,inplace=True)
merged_data
merged_data.isna().sum()
merged_data.info()
len(merged_data.columns)
merged_data.describe().T
merged_data['Score_15'].mean()
merged_data
merged_data.fillna(merged_data.mean(),inplace=True)
merged_data
merged_data['Region_x'].unique()
len(merged_data['Region_x'].unique())
new_happ_2015
correlation_15=new_happ_2015.corr()
correlation_15
# mutlulugu en fazla etkileyen degisken saglik degiskeni
correlation_16=new_happ_2016.corr()
correlation_16
# mutlulugu en fazla etkileyen degisken saglik degiskeni
correlation_17=new_happ_2017.corr()
correlation_17
# mutlulugu en fazla etkileyen degisken saglik degiskeni
correlation_18=new_happ_2018.corr()
correlation_18
# mutlulugu en fazla etkileyen degisken saglik degiskeni
correlation_19=new_happ_2019.corr()
correlation_19
# mutlulugu en fazla etkileyen degisken saglik degiskeni
merged_data['Country'].groupby(merged_data['Region_x']).count()
score=merged_data[['Country','Region_x','Score_15','Score_16','Score_17','Score_18','Score_19']]
score
score.mean(axis=1)
score['mean']=score.mean(axis=1)
score
sorted_score=score.sort_values(by='mean', ascending=False)
sorted_score
sorted_score.head(3)
sorted_score.tail(3)
region_score=sorted_score.groupby(sorted_score['Region_x']).mean()
region_score
corruption=merged_data[['Country','Region_x','Corruption_15','Corruption_16','Corruption_17','Corruption_18','Corruption_19']]
corruption
corruption['mean']=corruption.mean(axis=1)
corruption
sorted_corruption=corruption.sort_values(by='mean', ascending=False)
sorted_corruption
sorted_corruption.head(3)
sorted_corruption.tail(3)
region_corruption=sorted_corruption.groupby(sorted_corruption['Region_x']).mean()
region_corruption
freedom=merged_data[['Country','Region_x','Freedom_15','Freedom_16','Freedom_17','Freedom_18','Freedom_19']]
freedom
freedom['mean']=freedom.mean(axis=1)
freedom
region_freedom=freedom.groupby(freedom['Region_x']).mean()
region_freedom
sorted_region_freedom=region_freedom.sort_values(by='mean', ascending=False)
sorted_region_freedom
health=merged_data[['Country','Region_x','Health_15','Health_16','Health_17','Health_18','Health_19']]
health
health['mean']=health.mean(axis=1)
health
region_health=health.groupby(health['Region_x']).mean()
region_health
sorted_region_health=region_health.sort_values(by='mean', ascending=False)
sorted_region_health