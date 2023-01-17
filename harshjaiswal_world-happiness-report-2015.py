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
world_happiness_2015=pd.read_csv('../input/world-happiness/2015.csv')
world_happiness_2016=pd.read_csv('../input/world-happiness/2016.csv')
world_happiness_2017=pd.read_csv('../input/world-happiness/2017.csv')
world_happiness_2018=pd.read_csv('../input/world-happiness/2018.csv')
world_happiness_2019=pd.read_csv('../input/world-happiness/2019.csv')
world_happiness_2015.where(world_happiness_2015.Region=='Latin America and Caribbean').dropna()
bins=[0,15,100,178]
label=['Top 15 countries','countries up till 100th rank','lowestranking_country']
world_happiness_2015['binned'] = pd.cut(world_happiness_2015['Happiness Rank'], bins=bins, labels=label)
import matplotlib.pyplot as plt
import seaborn as sns
world_happiness_2015.rename(columns={'Happiness Score':'Score_Happiness','Happiness Rank':'Rank',
                                     'Economy (GDP per Capita)':'Gdp','Health (Life Expectancy)':'Lfeexp',
                                     'Trust (Government Corruption)':'Truth','Dystopia Residual':'dystopiaResidual'},inplace=True)



General_correlation=world_happiness_2015.copy()
General_correlation=General_correlation.drop(['Region','Standard Error','Rank'],axis=1)
General_correlation=General_correlation.corr(method ='pearson')
plt.figure(figsize=(8,8))
sns.heatmap(General_correlation,annot=True,cmap='gnuplot_r').set_title('Factors Governing Rank');
north_america=world_happiness_2015.where(world_happiness_2015.Region=='North America')
north_america=north_america.dropna()
north_america=north_america.drop(['Region','Rank','binned','Standard Error'],axis=1)
north_america=north_america.corr(method ='pearson')
plt.figure(figsize=(10,10))
plt.title('Region:North America')
sns.heatmap(north_america,annot=True,cmap='BuGn');
western_europe=world_happiness_2015.where(world_happiness_2015.Region=='Western Europe')
western_europe=western_europe.drop('binned',axis=1)
western_europe=western_europe.dropna()
western_europe=western_europe.drop(['Rank','Region','Country','Standard Error'],axis=1)
western_europe=western_europe.corr(method ='pearson')
plt.figure(figsize=(10,10))
plt.title('Western-Europe')
sns.heatmap(western_europe,annot=True,cmap="YlGnBu");
Australia_Newzeland=world_happiness_2015.where(world_happiness_2015.Region=='Australia and New Zealand')
Australia_Newzeland=Australia_Newzeland.dropna()
Australia_Newzeland=Australia_Newzeland.corr(method ='pearson')
plt.figure(figsize=(10,10))
plt.title('Australia and New-Zeland')
sns.heatmap(Australia_Newzeland,annot=True,cmap='winter');
Eastern_Asia=world_happiness_2015.where(world_happiness_2015.Region=='Eastern Asia')
Eastern_Asia=world_happiness_2015.dropna()
Eastern_Asia=Eastern_Asia.drop(['Rank','Standard Error'],axis=1)
Eastern_Asia=Eastern_Asia.corr(method ='pearson')
plt.figure(figsize=(10,10))
plt.title('Eastern Asia')
sns.heatmap(Eastern_Asia,annot=True);
Central_Eastern_Europe=world_happiness_2015.where(world_happiness_2015.Region=='Central and Eastern Europe')
Central_Eastern_Europe=Central_Eastern_Europe.dropna()
Central_Eastern_Europe=Central_Eastern_Europe.drop(['Rank','binned','Standard Error'],axis=1)
Central_Eastern_Europe=Central_Eastern_Europe.corr(method ='pearson')
plt.figure(figsize=(10,10))
plt.title('Central Eastern Europe')
sns.heatmap(Central_Eastern_Europe, annot=True,cmap='OrRd_r');
SouthernAsia=world_happiness_2015.where(world_happiness_2015.Region=='Southern Asia')
SouthernAsia=SouthernAsia.dropna()
SouthernAsia=SouthernAsia.drop(['binned','Rank','Region','Standard Error'],axis=1)
SouthernAsia=SouthernAsia.corr(method ='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(SouthernAsia,annot=True,cmap='mako').set_title('Southern Asia');
southeasternasia=world_happiness_2015.where(world_happiness_2015.Region=='Southeastern Asia')
southeasternasia=southeasternasia.dropna()
southeasternasia=southeasternasia.drop(['binned','Rank','Region','Standard Error'],axis=1)
southeasternasia=southeasternasia.corr(method ='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(southeasternasia,annot=True,cmap="viridis").set_title('South East Asia');
subsharaafrica=world_happiness_2015.where(world_happiness_2015.Region=='Sub-Saharan Africa')
subsharaafrica=subsharaafrica.dropna()
subsharaafrica=subsharaafrica.drop(['binned','Rank','Region','Standard Error'],axis=1)
subsharaafrica=subsharaafrica.corr(method ='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(subsharaafrica,annot=True,cmap='rocket_r').set_title('Sub Shara Africa');
Middle_east_africa=world_happiness_2015.where(world_happiness_2015.Region=='Middle East and Northern Africa')
Middle_east_africa=Middle_east_africa.dropna()
Middle_east_africa=Middle_east_africa.drop(['binned','Rank','Region','Standard Error'],axis=1)
Middle_east_africa=Middle_east_africa.corr(method ='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(Middle_east_africa,annot=True,cmap='flag').set_title('Middle East  and Northern Africa');
Latin_america=world_happiness_2015.where(world_happiness_2015.Region=='Latin America and Caribbean')
Latin_america=Latin_america.dropna()
Latin_america=Latin_america.drop(['binned','Rank','Region','Standard Error'],axis=1)
Latin_america=Latin_america.corr(method ='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(Latin_america,annot=True,cmap="cubehelix").set_title("Latin America and Caribbean");
 
world_happiness_2015.sort_values('Gdp',ascending=False,inplace=True)
highest_economy=world_happiness_2015
highest_economy=highest_economy[['Country','Gdp','Region']].head(10)
plt.figure(figsize=(8,8))
plt.title('Highest Econmy 2015')
sns.barplot(highest_economy.Gdp,highest_economy.Country,hue=highest_economy.Region).legend(loc='best',bbox_to_anchor=(1.8, 0.5), ncol=1);
world_happiness_2015.sort_values('Family',ascending=False,inplace=True)
world_happiness_2015[['Country','Family']].head(10)
plt.figure(figsize=(10,10))
plt.title("Family Satisfaction")
sns.pointplot(world_happiness_2015.Country.head(10),world_happiness_2015.Family.head(10),data=world_happiness_2015,hue=world_happiness_2015.Region).legend(loc='best',bbox_to_anchor=(1.8, 0.5), ncol=1).set_title('Life expectancy');

world_happiness_2015.sort_values('Truth',ascending=False,inplace=True)
world_happiness_2015[['Country','Truth']].head(15)
plt.figure(figsize=(10,10))
plt.title('Trust in the Government')
sns.pointplot(world_happiness_2015.Truth.head(15),world_happiness_2015.Country.head(15),data=world_happiness_2015,hue=world_happiness_2015.Region).legend(loc='best',bbox_to_anchor=(1.8, 0.5), ncol=1).set_title('Trust In Government');
world_happiness_2015.sort_values('Freedom',ascending=False,inplace=True)
world_happiness_2015[['Country','Freedom']].head(10)
world_happiness_2015.sort_values('Freedom',ascending=False,inplace=True)
freedom=world_happiness_2015[['Country','Freedom']].head(10)
plt.figure(figsize=(7,7))
plt.title('Freedom')
sns.pointplot(world_happiness_2015.Freedom.head(10),world_happiness_2015.Country.head(10),data=world_happiness_2015,hue=world_happiness_2015.Region).legend(loc='best',bbox_to_anchor=(1.8, 0.5), ncol=1).set_title('Freedom');
world_happiness_2015.sort_values('Generosity',ascending=False,inplace=True)
world_happiness_2015[['Country','Generosity']].head(10)
plt.figure(figsize=(7,7))
sns.pointplot(world_happiness_2015.Generosity.head(15),world_happiness_2015.Country.head(15),data=world_happiness_2015,hue=world_happiness_2015.Region).legend(loc='best',bbox_to_anchor=(1.8, 0.5), ncol=1).set_title('Freedom');
world_happiness_2015.sort_values('dystopiaResidual',ascending=False,inplace=True)
world_happiness_2015[['Country','dystopiaResidual']].head(10)

plt.figure(figsize=(7,7))
sns.pointplot(world_happiness_2015.dystopiaResidual.head(15),world_happiness_2015.Country.head(15),data=world_happiness_2015,hue=world_happiness_2015.Region).legend(loc='best',bbox_to_anchor=(1.8, 0.5), ncol=1).set_title('dystopian Residual');