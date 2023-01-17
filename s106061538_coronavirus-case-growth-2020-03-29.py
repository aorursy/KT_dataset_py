import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib 

matplotlib.style.use('ggplot')

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
### Data cleaning



def drop_duplicates(confirmed):



    '''

    groups = confirmed.groupby(['Lat','Long']).count() # location double index

    new = confirmed.set_index(['Lat','Long']) # set index to delete

    index_drop = new.loc[new.index.isin(groups[groups['1/22/20']>1].index)].sort_index().index # find location double index -> currently cruise

    new = new.loc[~new.index.isin(index_drop)].reset_index() # drop index

    '''

    

    # simpler solution

    new = confirmed[~confirmed.duplicated(subset=['Lat','Long'], keep=False)]

    

    return new



# drop unreasonable value
### Data building



def createdaily(Confirmed):

    

    ### create daily cases from daily cumulated cases

    

    # preprocessing

    

    copy1 = Confirmed.drop(['Country/Region'], axis=1).to_numpy()

    datename = Confirmed.drop(['Country/Region'], axis=1).columns

    

    # create daily

    

    for i in range(len(copy1)): # region

        for j in reversed(range(1,len(copy1[0]))): # daily 

            copy1[i][j] = copy1[i][j] - copy1[i][j-1]

    

    # adding columns

    

    df = pd.DataFrame(copy1)

    df.columns = datename

    df['max_value'] = df.max(axis=1)

    df = pd.concat([Confirmed[['Country/Region']], df], axis=1)

    df = df.sort_values(df.columns[-1], ascending=False).drop(columns=['max_value'])         

    

    return df



def firstdaymatch(Confirmed):

    

    ### fitting the start day of each country

    

    # preprocessing

    

    copy1 = Confirmed.drop(['Country/Region'], axis=1).to_numpy()

    datename = Confirmed.drop(['Country/Region'], axis=1).columns

    

    # shift to the start day

    

    for i in range(len(copy1)):  # region

        if copy1[i][0] == 0:

            for j in range(len(copy1[0])): # days

                if copy1[i][j] > 0: 

                    copy1[i] = np.roll(copy1[i],-j) # roll

                    j = len(copy1[0]) # stop pointing

    

    # adding columns

    

    df = pd.DataFrame(copy1)

    df[df.columns] = df[df.columns].replace({0:np.nan})

    #df.columns = datename

    #df[datename] = df[datename].replace({0:np.nan}) # fill 0 to nan

    df['max_value'] = df.max(axis=1)

    df = pd.concat([Confirmed[['Country/Region']], df], axis=1)

    df = df.sort_values(df.columns[-1], ascending=False).drop(columns=['max_value'])                

                

    return df
confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')



dailyadded = drop_duplicates(confirmed).drop(columns=['Lat','Long']).groupby('Country/Region').sum().sort_values(confirmed.columns[-1],ascending=False).reset_index()

daily = createdaily(dailyadded)

dailyadded_1dM = firstdaymatch(dailyadded)

daily_1dM = firstdaymatch(daily)
ax = dailyadded.set_index('Country/Region').transpose().plot(figsize=(10,6))

ax.set(xlabel="Day")

ax.set(ylabel='Cases')

_ = ax.legend(bbox_to_anchor=(1,1))
ax = daily.set_index('Country/Region').transpose().plot(figsize=(10,6))

ax.set(xlabel="Day")

ax.set(ylabel='Cases')

_ = ax.legend(bbox_to_anchor=(1,1))
ax = dailyadded_1dM.set_index('Country/Region').transpose().plot(figsize=(10,6))

ax.set(xlabel="Day")

ax.set(ylabel='Cases')

_ = ax.legend(bbox_to_anchor=(1,1))
ax = daily_1dM.set_index('Country/Region').transpose().plot(figsize=(10,6))

ax.set(xlabel="Day")

ax.set(ylabel='Cases')

_ = ax.legend(bbox_to_anchor=(1,1))