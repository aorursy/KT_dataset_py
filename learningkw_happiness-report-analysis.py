import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline 
happiness2017 = pd.read_csv('2017.csv')
happiness2017.shape
happiness2017.head()
#replaced white spaces with underscore 
happiness2017.columns = happiness2017.columns.str.lower()
happiness2017.head()
renaming = {'happiness.rank':"rank","happiness.score":"score",'economy..gdp.per.capita.':'gdp_capita',
            'health..life.expectancy.':'health','trust..government.corruption.':'trust'}
happiness2017.rename(renaming,axis =1,inplace=True)
happiness2017.head()
happiness2017.dtypes
happiness2017.isnull().sum()
happiness2017.describe()
happiness2017[(happiness2017['gdp_capita']==0) | (happiness2017['family']==0)|
              (happiness2017['health']==0) | (happiness2017['freedom']==0)|
              (happiness2017['generosity']==0) | (happiness2017['trust']==0)]

happiness2017.drop(['whisker.high','whisker.low','dystopia.residual'],inplace=True, axis = 1)
fig = plt.figure(figsize=(10,10))

num = 1
for label in happiness2017.columns[3:9]: 
    ax = fig.add_subplot(2,3,num)
    plt.scatter(happiness2017['score'],happiness2017[label])
    plt.title(label,loc='center')
    num +=1
plt.show()
correlation=happiness2017.corr()
correlation
correlation['score']
#for practice purposes, I also visualize the corrlation with heatmap. 
figure = plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot = True)
happiness2015 = pd.read_csv('2015.csv')
happiness2015.shape
happiness2015.head()
check = happiness2017['country'].isin(happiness2015['Country'])
happiness2017[~check]
happiness2015.columns = happiness2015.columns.str.lower()
happiness2015[['country','region']].head()
happiness2017_region = pd.merge(happiness2017,happiness2015[['country','region']], on='country')
region_score = happiness2017_region.groupby('region',as_index=False)['score'].mean().sort_values(ascending=False,by='score')
region_score
region_score.plot.barh(x='region',y='score',figsize=(10,10))
happiness2017_avg = happiness2017_region.groupby('region',as_index=False).mean()
happiness2017_avg = happiness2017_avg.sort_values(by='score',ascending=False).set_index('region')
happiness2017_avg
figure = figure = plt.figure(figsize=(10,10))
sns.heatmap(happiness2017_avg[['gdp_capita','family','health','freedom','generosity','trust']], annot=True)