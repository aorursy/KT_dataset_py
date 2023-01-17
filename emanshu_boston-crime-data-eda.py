# import dependencies

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/crime.csv',encoding='ISO-8859-1')



df.head()
print(df.info())

print(' ')

print(df.describe())

print(' ')

print(df.shape)

print(' ')

print(df.columns)

print(' ')

print(df.dtypes)
figure = plt.figure(figsize=(13,6))

sns.heatmap(df.isnull(),yticklabels='')

plt.title('Null Values')



# plt.savefig('images/nullValueHeatMap.png',bbox_inches = 'tight')
# DROP NULL VALUES ??????
# create DF of unique offense and how many times they occured. 

offenseCounts = pd.DataFrame(df['OFFENSE_CODE_GROUP'].value_counts()).reset_index()

# rename columns

offenseCounts.rename(columns={'index':'Offense','OFFENSE_CODE_GROUP':'Count'}, inplace=True)

offenseCounts.head(10)
figure = plt.figure(figsize=(10,15))

sns.barplot(x=offenseCounts['Count'],y=offenseCounts['Offense'],palette='Set3')

plt.ylabel('Offense')

plt.xlabel('Count')

sns.despine()

# plt.savefig('images/offenseCount.png',bbox_inches = 'tight')
crimeAreas = pd.DataFrame(df['STREET'].value_counts()).reset_index()

crimeAreas.rename(columns={'index':'Street','STREET':'Count'},inplace=True)

crimeAreas.head(10)
figure = plt.figure(figsize=(10,10))



sns.barplot(x=crimeAreas['Count'].iloc[:15],y=crimeAreas['Street'].iloc[:15],palette='Set3')

plt.ylabel('Street')

plt.xlabel('Count')

plt.title('Top Ten Offense Streets')

sns.despine()

# plt.savefig('images/streetsOffenseCount.png',bbox_inches = 'tight')
districtCrimeRate = pd.DataFrame(df['DISTRICT'].value_counts()).reset_index()

districtCrimeRate.rename(columns={'index':'District','DISTRICT':'Count'},inplace=True)

districtCrimeRate.head(10)
figure = plt.figure(figsize=(10,10))



sns.barplot(x=districtCrimeRate['Count'].iloc[:15],y=districtCrimeRate['District'].iloc[:15],palette='Set3')

plt.ylabel('District')

plt.xlabel('Count')

plt.title('Districts with the most commited Offenses')

sns.despine()

# plt.savefig('images/districtOffenseCount.png',bbox_inches = 'tight')
offenseByDistrict = pd.DataFrame(df.groupby(['DISTRICT','OFFENSE_CODE_GROUP'])['OFFENSE_CODE_GROUP'].count())



offenseByDistrict.rename(columns={'OFFENSE_CODE_GROUP':'Count'},inplace=True)



offenseByDistrict.sort_values(['DISTRICT','Count'],ascending=False,inplace=True)



offenseByDistrict.head()
# grab list of top ten offense from offenseCounts table

topTenOffensesList = (offenseCounts['Offense'].loc[:9]).tolist()

print(topTenOffensesList)



# reset index of offenseByDistrict Table 

offenseByDistrictPlot = offenseByDistrict.reset_index()

# filter table by top ten offenses

offenseByDistrictPlot = pd.DataFrame(offenseByDistrictPlot[offenseByDistrictPlot['OFFENSE_CODE_GROUP'].isin(topTenOffensesList)])



offenseByDistrictPlot.head()
sns.set_context('paper',font_scale=2.5)

sns.set_style('white')

figure = plt.figure(figsize=(20,10))



sns.barplot(x=offenseByDistrictPlot['DISTRICT'],y=offenseByDistrictPlot['Count'],

            hue=offenseByDistrictPlot['OFFENSE_CODE_GROUP'],edgecolor='black',linewidth=.1,palette='Set3')

plt.legend(bbox_to_anchor=(1,1), loc='best', borderaxespad=1.1)

plt.title('Overall Top Offenses commited by District')

# plt.ylabel('Count',{'fontsize':22})

# plt.xlabel('District',{'fontsize':22})



# plt.savefig('images/offensesByDistrict.png',bbox_inches = 'tight')
offenseHeatDf = offenseByDistrictPlot.pivot(index='OFFENSE_CODE_GROUP',columns='DISTRICT',values='Count')

offenseHeatDf
figure = plt.figure(figsize=(15,10))



sns.heatmap(offenseHeatDf,linecolor='white',linewidth=.1,cmap='magma',)

plt.ylabel(' ')

plt.ylabel(' ')

plt.title('Heatmap of Offenses Commited by District')

# plt.savefig('images/districtsOffenseHeatMap.png',bbox_inches='tight')
byDayDF = pd.DataFrame(df.groupby(['DAY_OF_WEEK'])['OFFENSE_CODE_GROUP'].count()).reset_index().rename(columns={'DAY_OF_WEEK':'Day','OFFENSE_CODE_GROUP':'Count'})



byDayDF['Day'] = pd.Categorical(byDayDF['Day'], categories=

    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],

    ordered=True)



byDayDF.sort_values(by='Day')
sns.set()

sns.set_style('ticks')



figure = plt.figure(figsize=(15,2))

sns.barplot(x=byDayDF['Day'],y=byDayDF['Count'],palette='magma',)

sns.despine()

plt.ylabel(' ')

plt.title('Crimes Commited by Day')

# plt.savefig('images/crimesCountByDay.png')
crimeByDayCatg = pd.DataFrame(df.groupby(['DAY_OF_WEEK','OFFENSE_CODE_GROUP'])['OFFENSE_CODE_GROUP'].count())



crimeByDayCatg.rename(columns={'OFFENSE_CODE_GROUP':'Count'},inplace=True)



crimeByDayCatg.reset_index(inplace=True)





crimeByDayCatg['DAY_OF_WEEK'] = pd.Categorical(crimeByDayCatg['DAY_OF_WEEK'], categories=

    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],

    ordered=True)



crimeByDayCatg.sort_values(by='DAY_OF_WEEK')



# filter table to only include the overall top ten offenses. 

crimeByDayCatgHeatTopTen = pd.DataFrame(crimeByDayCatg[crimeByDayCatg['OFFENSE_CODE_GROUP'].isin(topTenOffensesList)])



# pivot table to create heat map

crimeByDayCatgHeat = crimeByDayCatgHeatTopTen.pivot(index='OFFENSE_CODE_GROUP',columns='DAY_OF_WEEK',values='Count').fillna(0)



crimeByDayCatgHeat.head()
sns.set()

sns.set_context('paper',font_scale=1.5)



figure = plt.figure(figsize=(10,7))

sns.heatmap(crimeByDayCatgHeat,linecolor='white',linewidths=1,cmap='magma',)



plt.ylabel(' ')

plt.xlabel(' ')

plt.xticks(rotation=30)

plt.title('Heatmap of Top Ten Offenses by Weekday')

# plt.savefig('images/offenseByWeekdayHeatmap.png')
offensesByYear = pd.DataFrame(df.groupby(['YEAR'])['OFFENSE_CODE_GROUP'].count()).reset_index().rename(columns={'OFFENSE_CODE_GROUP':'Count'})

offensesByYear



sns.set()

sns.set_style('white')



figure = plt.figure(figsize=(12,3))



sns.barplot(x=offensesByYear['YEAR'],y=offensesByYear['Count'],palette='magma')

sns.despine()

plt.ylabel(' ')

plt.xlabel(' ')

plt.title('Total Crimes Commited by Year')



# plt.savefig('images/crimesByyear.png')
# create list of unique years to use later 

years = df['YEAR'].unique().tolist()

print(years)



# create df that is grouped by year and offense code and tallys up the distinct offenses

offensesByYearCatg = pd.DataFrame(df.groupby(['YEAR','OFFENSE_CODE_GROUP'])[['OFFENSE_CODE_GROUP']].count())

#  rename second offense_code_group column to 'Count'

offensesByYearCatg.columns = ['Count']



#  reset index to set all columns on same level, makes it easier to access and filter by later on. 

offensesByYearCatg.reset_index(inplace=True)



#  order by the year and count columns from largest to smallest

offensesByYearCatg = offensesByYearCatg.sort_values(by=['YEAR','Count'],ascending=False)



offensesByYearCatg.head()
#  create blank data frame

topTenOffensesbyYear = pd.DataFrame()



# loop through years array created earlier and create df that only contains the top ten offenses commited per year

for x in range(len(years)):

    

#     as we loop through each year 'years[x]' 2017,2018 ect append the top ten offenses for each year. 



#     offensesByYearCatg[offensesByYearCatg['YEAR'] == 2018][:10]



    topTenOffensesbyYear = topTenOffensesbyYear.append(offensesByYearCatg[offensesByYearCatg['YEAR'] == years[x]][:10])

    

print(topTenOffensesbyYear.shape)

topTenOffensesbyYear
figure = plt.figure(figsize=(18,6))



sns.set_context('paper',font_scale=1.5,)

sns.barplot(x=topTenOffensesbyYear['YEAR'],y=topTenOffensesbyYear['Count'],hue=topTenOffensesbyYear['OFFENSE_CODE_GROUP'],

            palette='Set3')

# plt.xticks(rotation=88)

plt.legend(bbox_to_anchor=(1,1), loc='best', borderaxespad=1.1)

sns.despine()

plt.title('Top Ten Offenses by Year')

plt.ylabel('')

plt.xlabel('')