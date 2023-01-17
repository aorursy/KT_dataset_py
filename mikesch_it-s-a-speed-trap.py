import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')
df1 = pd.read_csv('../input/cameras.csv', header=0)
#find out type of data and number of values for each column

df1.info()
#converting date into time stamp

df1['DATE'] = pd.to_datetime(df1['DATE'])



#splitting day, month, and year

df1['DAY'] = [d.date().strftime('%d') for d in df1['DATE']]

df1['MONTH'] = [d.date().strftime('%m') for d in df1['DATE']]

df1['YEAR'] = [d.date().strftime('%Y') for d in df1['DATE']]



#adding column with day of week (0=Monday, 6=Sunday)

df1['DAY_OF_WEEK'] = df1['DATE'].dt.dayofweek



df1.head(3)
#dropping unneeded columns

df1.drop(['DATE', 'LOCATION'], axis=1, inplace=True)
#plotting average # of violations by month, split by year

years = ['2014','2015','2016']

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']



fig, ax = plt.subplots(1,3, figsize=(10,5), sharey=True)



for i in range(3):

    sns.barplot(df1.loc[df1['YEAR'] == years[i]]['MONTH'], 

                df1.loc[df1['YEAR'] == years[i]]['VIOLATIONS'], 

                errwidth=1, ax=ax[i]);

    ax[i].set_xticklabels(months, rotation=25);

    ax[i].set_xlabel(years[i]);

    ax[i].set_ylabel('');



ax[0].set_ylabel('Average Violations')

ax[0].set_xticklabels(months[6:], rotation=25);  #override xticklabels in 2014 as data starts in July
#creating crosstab, plotting months by year

df_ct = pd.crosstab(df1['MONTH'], df1['YEAR'])



for i in years:

    plt.plot(df_ct[i])



plt.xticks([a for a in range(13)], months, rotation=25);



plt.legend();
#plotting average # of violations by day of week, split by year



fig, ax = plt.subplots(1,3, figsize=(10,5), sharey=True)



for i in range(3):

    sns.barplot(df1.loc[df1['YEAR'] == years[i]]['DAY_OF_WEEK'], 

                df1.loc[df1['YEAR'] == years[i]]['VIOLATIONS'], 

                errwidth=1, ax=ax[i]);

    ax[i].set_xticklabels(['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'], rotation=25);

    ax[i].set_xlabel(years[i]);

    ax[i].set_ylabel('');



ax[0].set_ylabel('Average Violations');
#plotting average # of violations by CAMERA ID, split by year



fig, ax = plt.subplots(3,1, figsize=(12,10), sharex=True)



for i in range(3):

    sns.barplot(df1.loc[df1['YEAR'] == years[i]]['CAMERA ID'], 

                df1.loc[df1['YEAR'] == years[i]]['VIOLATIONS'], 

                errwidth=1, ax=ax[i]);

    ax[i].set_xticklabels(df1['CAMERA ID'].unique(), rotation=90, fontsize=5);

    ax[i].set_xlabel(years[i]);

    ax[i].set_ylabel('Average Violations');



ax[0].set_title('Average Annual Violations by Camera ID');
#setting up data to identify highest violations in 2016 by ADDRESS (which is same as CAMERA ID)

df_cam = df1.loc[df1['YEAR'] == '2016'][['ADDRESS','VIOLATIONS']]



#grouping & sorting data to rank # of violations by address

df_cam = df_cam.groupby('ADDRESS').sum().sort_values(by='VIOLATIONS', ascending=False)



#show percent of total violations that these 10 locations represent

perc_of_locations  = (len(df_cam.head(10)) / len(df_cam))*100

perc_of_violations = (sum(df_cam['VIOLATIONS'].head(10)) / sum(df_cam['VIOLATIONS']))*100



print('Top 10 locations are {:.0f}% of all locations & {:.0f}% of all violations'.format(perc_of_locations, perc_of_violations))



print('\nThe 10 locations to be (extra) careful near are:')

df_cam.head(10)