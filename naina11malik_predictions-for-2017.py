import seaborn as sns

import pandas as pd 

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/celebrity_deaths_4.csv',encoding='latin1')
data.sample(200)[['name','age','death_year','famous_for','cause_of_death','fame_score']].head()
deaths_per_month = data

normal_deaths_per_month = deaths_per_month[deaths_per_month.fame_score <20].groupby(['death_year','death_month']).count().age

famous_deaths_per_month = deaths_per_month[deaths_per_month.fame_score >=20].groupby(['death_year','death_month']).count().age
famous_deaths_per_month.index = range(len(famous_deaths_per_month))

normal_deaths_per_month.index = range(len(normal_deaths_per_month))
sns.set_style('dark',{"axes.facecolor": 'ghostwhite'})

sns.tsplot(normal_deaths_per_month,color=sns.color_palette("Greys_d",n_colors=1),condition='"Normal" Celebrities')

sns.tsplot(famous_deaths_per_month,color=sns.color_palette("Reds_d",n_colors=1),condition='Famous Celebrities')

plt.title('Celebrity deaths per month since Jan 2006')

plt.ylabel('Count')

plt.xlabel('Month')
model = sm.WLS(famous_deaths_per_month,famous_deaths_per_month.index)

results = model.fit()

print('Model R^2:',results.rsquared)
sns.set_style('dark',{"axes.facecolor": 'ghostwhite'})

sns.tsplot(famous_deaths_per_month,color=sns.color_palette("Reds_d",n_colors=1),condition='Famous Celebrities')

sns.tsplot(results.predict(range(len(famous_deaths_per_month)+12)),color=sns.color_palette("Greens_d",n_colors=1),condition='WLS model prediction')

plt.title('Celebrity deaths per month since Jan 2006')

plt.ylabel('Count')

plt.xlabel('Month')
print(sum(famous_deaths_per_month[len(famous_deaths_per_month)-12:len(famous_deaths_per_month)]),'famous celebrities died in 2016')
print(round(sum(results.predict([i for i in range(len(famous_deaths_per_month)+1,len(famous_deaths_per_month)+13)]))),'are expected to die in 2017')
plt.axhline(0,ls='--', color='black',lw=0.5)

plt.scatter(range(len(results.resid)),results.resid)

plt.title('Residuals')

print('Mean of model residuals:',round(np.mean(results.resid),2))