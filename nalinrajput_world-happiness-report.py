#installing the jovian library
!pip install jovian
#importing the libraries required for visualisation
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import jovian
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#loading the datasets y1,y2,y3 corresponding to 2017,2015 and 2019
y1_df=pd.read_csv('/kaggle/input/world-happiness/2017.csv',index_col=1)
y2_df=pd.read_csv('/kaggle/input/world-happiness/2015.csv',index_col=2)
y3_df=pd.read_csv('/kaggle/input/world-happiness/2019.csv',index_col=0)
y1_df.head()
y2_df.head()
y3_df.head()
y1_df.drop(['Whisker.low','Whisker.high'],axis=1,inplace=True)
#renaming columns in y1 for convenience
y1_col={'Country':'Country','Happiness.Rank':'rank','Happiness.Score':'Score','Economy..GDP.per.Capita.':'GDP_per_capita','Family':'Family','Health..Life.Expectancy.':'life_expectancy','Freedom':'Freedom','Generosity':'Generosity','Trust..Government.Corruption.':'corr_perception','Dystopia.Residual':'dystopia_residual'}
y1_df.rename(columns=y1_col,inplace=True)
#renaming in y2
y2_df.drop(['Standard Error'],axis=1,inplace=True)
y2_col={'Country':'Country','Happiness Score':'Score','Economy (GDP per capita)':'GDP_per_capita','Family':'Social support','Health (Life Expectancy)':'life_expectancy','Freedom':'Freedom','Trust (Government corruption)':'corr_perception','Generosity':'Generosity','Dystopia Residual':'dystopia_residual'}
y2_df.rename(columns=y2_col,inplace=True)
#renaming in y3
y3_col={'Overall rank':'rank','Country or region':'Country','Score':'Score','GDP per capita':'GDP_per_capita','Social support':'Social support','Healthy life expectancy':'life_expectancy','Freedom to make life choices':'Freedom','Generosity':'Generosity','Perceptions of corruption':'corr_perception'}
y3_df.rename(columns=y3_col,inplace=True)
#setting a darkgrid style for each visualisation
sns.set_style("darkgrid")
project_name='World Happiness Report'
#for 2015 and 2017
plt.figure(figsize=(10,6))
a=10
plt.hist(y2_df.Score,a,label='2015',alpha=0.3,color='red')
plt.hist(y1_df.Score,a,label='2017',alpha=0.5,color='skyblue')
plt.ylabel('No. of countries',size=13)
plt.legend(loc='upper right')
plt.title("Distribution of Happiness scores across 2015,2017",size=16)
#for 2017 and 2019
plt.figure(figsize=(10,6))
b=10
plt.hist(y1_df.Score,b,label='2017',alpha=0.3)
plt.hist(y3_df.Score,b,label='2019',alpha=0.3)
plt.ylabel("No. of Countries",size=13)
plt.legend(loc="upper right")
plt.title('Distribution of Happiness scores across 2017,2019',size=16)
#correlation values for 2015 dataset
#creating a copy of the dataset with 4 columns.
y2=y2_df.copy()
y2.drop(['Social support','life_expectancy','Generosity','dystopia_residual'],axis=1,inplace=True)

#creating a correlation matrix between numeric columns
c2=y2.corr(method='pearson')
plt.figure(figsize=(10,6))
sns.heatmap(c2,annot=True)
#correlations for 2017 dataset
y1=y1_df.copy()
y1.drop(['Family','life_expectancy','Generosity','dystopia_residual'],axis=1,inplace=True)

c1=y1.corr(method='pearson')
plt.figure(figsize=(10,6))
sns.heatmap(c1,annot=True,cmap="YlOrRd")
#correlations for 2019 dataset
y3=y3_df.copy()
y3.drop(['Social support',
       'life_expectancy', 'Generosity'],axis=1,inplace=True)
c3=y3.corr()
plt.figure(figsize=(10,6))
sns.heatmap(c3,annot=True,cmap='PuBuGn')
#creating new datasets comprising of below mentioned columns
x1=y1_df[['Generosity','Family','Score']].copy()
x2=y2_df[['Generosity','Social support','Score']].copy()
x3=y3_df[['Generosity','Social support','Score']].copy()
#for year 2015
a2=x2.corr()
plt.figure(figsize=(10,6))
sns.heatmap(a2,annot=True)
#for year 2017
a1=x1.corr()
plt.figure(figsize=(10,6))
sns.heatmap(a1,annot=True,cmap='GnBu')
#for year 2019
a3=x3.corr()
plt.figure(figsize=(10,6))
sns.heatmap(a3,annot=True,cmap='RdPu')
#between GDP_per_capita and Happiness Score for y1_df(2017 dataset)
plt.figure(figsize=(10,6))
sns.pairplot(y1)

plt.figure(figsize=(10,6))
sns.pairplot(y2)
plt.figure(figsize=(10,6))
sns.pairplot(y3)
plt.figure(figsize=(10,6))
a=10
plt.hist(y2_df.life_expectancy,a,label='2015',alpha=0.3,color='red')
plt.hist(y1_df.life_expectancy,a,label='2017',alpha=0.5,color='skyblue')
plt.ylabel('No. of countries',size=13)
plt.legend(loc='upper right')
plt.title("Satisfaction with Life expectancy across 2015,2017",size=16)
plt.figure(figsize=(10,6))
a=10
plt.hist(y1_df.life_expectancy,a,label='2017',alpha=0.3)
plt.hist(y3_df.life_expectancy,a,label='2019',alpha=0.5)
plt.ylabel('No. of countries',size=13)
plt.legend(loc='upper right')
plt.title("Satisfaction with Life expectancy across 2017,2019",size=16)
#sorting values in 2017 dataset with ascending values of Life expectancy.
y1_df.sort_values('life_expectancy',axis=0,ascending=True)
y1_df.sort_values('life_expectancy',axis=0,ascending=False)
#here is the 2015 dataset
y2_df.head(5)
#creating a new series consisting of mean of happiness scores taken across different regions as specified.
#Converting this series into a dataframe
region=y2_df.groupby(['Region']).Score.mean()
region_df=pd.DataFrame(data=region)
reg=region_df.sort_values(by='Score',ascending=False,axis=0)
reg
plt.figure(figsize=(10,7))
plt.title('Happiness Scores across different regions')
sns.barplot(x='Score',y=reg.index,data=reg,palette='mako')
#creating a new dataset from y2_df comprising of means of gdp_per_Capita score per region.
gdpc=y2_df.groupby(['Region'])['Economy (GDP per Capita)'].mean()
gdpc_df=pd.DataFrame(data=gdpc)
gdp=gdpc_df.sort_values(by='Economy (GDP per Capita)',ascending=False,axis=0)
gdp
plt.figure(figsize=(10,7))
plt.title('Satisfaction with gdp in different regions')
sns.barplot(x='Economy (GDP per Capita)',y=gdp.index,data=gdp,palette='rocket')
trust=y2_df.groupby(['Region'])['Trust (Government Corruption)'].mean()
trust_df=pd.DataFrame(data=trust)
tru=trust_df.sort_values(by='Trust (Government Corruption)',ascending=False,axis=0)
tru
plt.figure(figsize=(10,7))
plt.title('Satisfaction with governments in different regions')
sns.barplot(x='Trust (Government Corruption)',y=tru.index,data=tru,palette='viridis')
freedom=y2_df.groupby(['Region'])['Freedom'].mean()
freedom_df=pd.DataFrame(data=freedom)
free=freedom_df.sort_values(by='Freedom',ascending=False,axis=0)
free
plt.figure(figsize=(10,7))
plt.title('Freedom score across different regions')
sns.barplot(x='Freedom',y=free.index,data=free,palette='rocket_r')