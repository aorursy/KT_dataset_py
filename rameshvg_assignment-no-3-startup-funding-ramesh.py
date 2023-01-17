# Introduction:
# This dataset contains information on startup name,industry vertical,location,investors of the startup etc.
# load the dataset and start analysing....
import pandas as pd
import numpy as np
%matplotlib inline
import os
print(os.listdir("../input"))                 
star = pd.read_csv("../input/startup_funding.csv") 
star.head()
star.info()
# 10 variables and 2372 observations..
# but few columns have a lot of na values...
#so lets start cleaning the dataset and analyse the percentage of na's in the dataset..
star._get_numeric_data().columns
star.describe(include=object)
nas=pd.isnull(star).sum()
nas
percent_nas = nas/star.shape[0] * 100
percent_nas
# the subvertical and Amountinusd columns have high %tage of na's 39 and 35 % respectively..
!pip install missingno
import missingno
missingno.matrix(star)
star['AmountInUSD'].head()
## Converting the String of numbers into float object through string methods and droping na values also..
star['AmountInUSD']=star['AmountInUSD'].dropna().apply(lambda rstr: float(rstr.replace(',','')))
star['AmountInUSD'].sum()        ### the Total Startup investment 18.35 us billion....
star.info()
star.describe()
star['AmountInUSD'].plot.box(figsize=(10,5))
q3 = star['AmountInUSD'].quantile(0.75)
q1 = star['AmountInUSD'].quantile(0.25)
iqr = q3 - q1
iqr
lw=q1 - 1.5 * iqr
uw=q3 + 1.5 * iqr
print(lw,uw)
# Possible oultliers above 14445000..its debatable whether it can be considered or not...
#Analysing one column at a time...
##  Checking for frequency distribution in different industry Verticals...Categorical Column..
freq=star['IndustryVertical'].value_counts().head(10)
freq
freq.plot.bar()
##ploting the frequcncy distribution...AM considering the top 10 verticals only ..
# lots of new startups have come up in Consumer Internet, Technology and Ecommerce Verticals...
##So takeaway: Consumer Internet industry vertical is rocking the startup space...

## Trying to analyse the LOcation column to check where the investments are pouring in...Location col...
freq_loc=star['CityLocation'].value_counts().head(10)
freq_loc
percent=freq_loc/star.shape[0] * 100
percent
freq_loc.plot.bar()
#percent.plot.bar()
#The Top 3 investments are happening in Bangalore , Mumbai and New Delhi..with 27% , 19% and 16% respectively..
##So takeaway: Bangalore is the best city for Startup Companies

## Basically we are trying to analyse the two columns here...One can be numercial and other can be Categorical col...
aa=star.boxplot(column='AmountInUSD', by='IndustryVertical',figsize=(30,15), rot=10)
aa.grid(False)
a=star.boxplot(column='AmountInUSD', by='CityLocation',figsize=(20,10), rot=10)
a.grid(False)
star.groupby('IndustryVertical')['AmountInUSD'].count().plot.bar(figsize=(20,10))
star.groupby('CityLocation')['AmountInUSD'].count().plot.bar(figsize=(20,10))
## The above plots show the amount invested across different verticals 
## The above plot shows the investment across different locations....
funding_type=star.groupby('InvestmentType')['AmountInUSD'].count()
funding_type
funding_type.plot.bar()
## The major funding is happening through Private Equity and Seed Funding...
p = star.boxplot(column='AmountInUSD', by='InvestmentType', 
               figsize=(20, 10), rot=10)
p.grid(False)
a=star.groupby('IndustryVertical')['AmountInUSD'].count()
a.sort_values(ascending=False).head()
### Applying T Test on one Categorical Column and one numerical coulumn...
# and considering the ConsumerInternet industry vertical and AmountInUSD col..
##IndustryVertical    VS     Amount Invested...
consumer = star[star['IndustryVertical'] == 'Consumer Internet']['AmountInUSD'].values
technology = star[star['IndustryVertical'] == 'Technology']['AmountInUSD'].values
print(len(consumer), len(technology))
from scipy.stats import ttest_ind
s, p = ttest_ind(consumer, technology)

if p > 0.05:
    print ('INDUSTRY VERTICAL does not influence Amount Invested')   #Accept the null hypothesis as p >0.05
else:
    print('INDUSTRY VERTICAL does influence Amount Invested')  #Reject the null hypothesis as p <0.05  and accept the alternate hypothesis..        
### Applying Annova on one Categorical Column and one numerical coulumn...
# and considering the ConsumerInternet industry vertical and AmountInUSD col..
from scipy.stats import f_oneway
s, p = f_oneway(consumer, technology)

if p > 0.05:
    print ('INDUSTRY VERTICAL does not influence Amount Invested')
else:
    print('INDUSTRY VERTICAL does influence Amount Invested')
# TAKEAWAY= The investment amount does depend on the Industry Vertical ...
# which has been proved by the above DATAS, PLOTS AND HYPOTHESIS      TESTS ALSO...


###  CORELATION  
star.info()
star.plot.scatter(x='SNo', y='AmountInUSD')
star.corr()
# Since there are no two numerical columns in the dataset we cannot do corelation ...
# as finding corelation between the SNo and Amount invested will not be ideal..

import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
p = sns.heatmap(star.corr(), cmap='Blues')
### not much of inference from the above plot as numerical cols are not present...

obs = star.groupby(['IndustryVertical', 'CityLocation']).size()
obs.name = 'Freq'
obs = obs.reset_index()

obs = obs.pivot_table(index='IndustryVertical', columns='CityLocation',
                values='Freq')
obs
sns.heatmap(obs, cmap='CMRmap_r')
obs = star.groupby(['IndustryVertical', 'InvestmentType']).size()
obs.name = 'Freq'
obs = obs.reset_index()

obs = obs.pivot_table(index='IndustryVertical', columns='InvestmentType',
                values='Freq')
obs
sns.heatmap(obs, cmap='CMRmap_r')
