import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

%matplotlib inline



state_data = '../input/State_time_series.csv'

df=pd.read_csv(state_data)

city_data = '../input/City_time_series.csv'

dfCity = pd.read_csv(city_data)

State_house = pd.read_csv("../input/State_time_series.csv", parse_dates=['Date'])



States = ['California','Colorado','Michigan']

newdf = df.loc[df['RegionName'].isin(States)]

newdf.Date = pd.to_datetime(newdf.Date)

newdf2 = newdf.loc[newdf['Date'].dt.year == 2016]

newdf3 = df.loc[df['RegionName'] == 'California']

newdf4 = df.loc[df['RegionName'] == 'Colorado']

newdf5 = df.loc[df['RegionName'] == 'Michigan']

newdf6 = dfCity.loc[dfCity['RegionName'] == 'san_franciscosan_franciscoca']

newdf6.Date = pd.to_datetime(newdf6.Date)

newdf7 = dfCity.loc[dfCity['RegionName'] == 'denverdenverco']

newdf7.Date = pd.to_datetime(newdf7.Date)

newdf8 = dfCity.loc[dfCity['RegionName'] == 'detroitwaynemi']

newdf8.Date = pd.to_datetime(newdf8.Date)
def plotDistribution(data,metric):

    """ Plot distributions """  

    sns.set_style("whitegrid")

    distributionTwo = sns.FacetGrid(data, hue='RegionName',aspect=2.5)

    distributionTwo.map(sns.kdeplot,metric,shade=True)

    distributionTwo.set(xlim=(100000, 550000))

    distributionTwo.add_legend()

    distributionTwo.set_axis_labels(str(metric), 'Proportion')

    distributionTwo.fig.suptitle(str(metric)+' vs Region (2016)')

plotDistribution(newdf2, 'MedianListingPrice_SingleFamilyResidence')

plotDistribution(newdf2, 'MedianSoldPrice_AllHomes')
def priceOverTime(data,label):

    """Plot price over time"""

    data.groupby(newdf.Date.dt.year)['MedianSoldPrice_AllHomes'].mean().plot(kind='bar', figsize=(10, 6), color='grey', edgecolor = 'black', linewidth = 2)

    plt.suptitle(label, fontsize=12)

    plt.ylabel('MedianSoldPrice_AllHomes')

    plt.xlabel('Year')

    plt.show()    

priceOverTime(newdf3, "California")

priceOverTime(newdf4, "Colorado")

priceOverTime(newdf5, "Michigan")
def priceOverTime2(data,label):

    data.groupby(data.Date.dt.year)['MedianSoldPrice_AllHomes'].mean().plot(kind='bar', figsize=(10, 6), color='grey', edgecolor = 'black', linewidth = 2)

    plt.suptitle(label, fontsize=12)

    plt.ylabel('MedianSoldPrice_AllHomes')

    plt.xlabel('Year')

    plt.show()

priceOverTime2(newdf6,'San Francisco')

priceOverTime2(newdf7,'Denver')

priceOverTime2(newdf8,'Detroit')
# The following code is a modified version of a snippet from https://www.kaggle.com/shelars1985/zillow-market-overview



State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['ZHVI_SingleFamilyResidence'].mean().unstack()

State_raw_house.columns.name = None      

State_raw_house = State_raw_house.reset_index()  

State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]

State_raw_house = State_raw_house.dropna()

Feature = State_raw_house['RegionName']

weightage = State_raw_house[2010]

total = State_raw_house[2017]

percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100

mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2

weightage = np.array(weightage)

Feature = np.array(Feature)

total = np.array(total)

percent = np.array(percent)

mid_pos  = np.array(mid_pos)



idx = percent.argsort()

Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]



s = 1

size=[]

for i, cn in enumerate(weightage):

     s = s + 1        

     size.append(s)

    

fig, ax = plt.subplots(figsize=(13, 13))

ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)

ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)

ax.set_xlabel('Home Value')

ax.set_ylabel('States')

ax.spines['right'].set_visible(False)

ax.grid()



for i, txt in enumerate(Feature):

      ax.annotate(txt, (720000,size[i]),fontsize=12,rotation=0,color='Red')

      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),

            arrowprops=dict(facecolor='LightGreen', shrink=0.06),

            )

for i, pct in enumerate(percent):

     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')



ax.annotate('2010 Home Value', (300000,26),fontsize=14,rotation=0,color='Green')

ax.annotate('2017 Home Value', (300000,25),fontsize=14,rotation=0,color='Blue');



ax.annotate('w/ Percent Change', (300000,24),fontsize=14,rotation=0,color='Brown');