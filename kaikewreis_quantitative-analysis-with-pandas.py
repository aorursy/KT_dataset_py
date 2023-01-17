import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline  
df = pd.read_csv('../input/the-game-awards/the_game_awards.csv', encoding='UTF-8')

winners = df.loc[df['winner'] == 1]
df.head(2)
df_winners_company = pd.DataFrame(df['company'].value_counts())

df_winners_company = df_winners_company.reset_index()

df_winners_company.columns = ['Company', 'Awards']
sns.set(style="whitegrid")

plot1 = sns.barplot(x='Company', y='Awards', data=df_winners_company[0:9]);

plot1.set_xticklabels(plot1.get_xticklabels(), rotation=45, ha='right');
df_winners_company[0:9]
for year in winners['year'].unique():

    # Get the data only for one year in specific

    winners_in_year = winners.loc[winners['year'] == year]

    # Get the value count for the companies in that year

    winners_companies = pd.DataFrame(winners_in_year['company'].value_counts())

    winners_companies = winners_companies.reset_index()

    winners_companies.columns = ['Company', 'Awards']

    # LOOP WITH CONDITIONAL - There is the same quantity of awards in the same year?

    for i in range(0,len(winners_companies)):

        # BREAK CONDITION - If we have one winner and the others companies has 

        # less than your awards then break break the loop

        if i != 0 and (winners_companies.loc[0,'Awards'] > winners_companies.loc[i,'Awards']):

            break

        # Print results

        print('In ',year,'the company that got most awards was ',winners_companies.loc[i,'Company'],' with ',winners_companies.loc[i,'Awards'],' awards.')
for year in winners['year'].unique():

    # Get the data only for one year in specific

    winners_in_year = winners.loc[winners['year'] == year]

    # Get the value count for the companies in that year

    winners_nominee = pd.DataFrame(winners_in_year['nominee'].value_counts())

    winners_nominee = winners_nominee.reset_index()

    winners_nominee.columns = ['Nominee', 'Awards']

    # LOOP WITH CONDITIONAL - There is the same quantity of awards in the same year?

    for i in range(0,len(winners_nominee)):

        # BREAK CONDITION - If we have one winner and the others companies has less than your awards then break break the loop

        if i != 0 and (winners_nominee.loc[0,'Awards'] > winners_nominee.loc[i,'Awards']):

            break

        # Print results

        print('In ',year,'the game that got most awards was ',winners_nominee.loc[i,'Nominee'],' with ',winners_nominee.loc[i,'Awards'],' awards.')
# Define previously the coluns manually

event = [2014,2014,2014,2015,2016,2017,2017,2018,2018,2019]

company = ['Ubisoft Montpellier','Nintendo EAD','BioWare','Nintendo','Blizzard Entertainment','StudioMDHR','Nintendo','Rockstar Games',

           'SIE Santa Monica Studio/Sony Interactive Entertainment','ZA/UM']

country = ['France','Japan','Canada','Japan','USA','Canada','Japan','USA','USA','England']

awards = [2,2,2,3,3,3,3,3,3,4]



# Create the dataframe

data = pd.DataFrame(list(zip(event,company,country,awards)),columns=['event','company','country','awards'])
data.groupby(['event', 'country']).size()