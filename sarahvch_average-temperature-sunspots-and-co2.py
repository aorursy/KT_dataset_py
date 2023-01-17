import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv")

ss = pd.read_csv("../input/yearly-mean-sunspot-numbers/sunspotnumber.csv")

cd = pd.read_csv("../input/carbon-dioxide/archive.csv")
df.head()
#clean up year

def to_year(date):

    """

    returns year from date time

    """

    for i in [date]:

        first = i.split('-')[0]

        return int(first)

 

df['year'] = df['dt'].apply(to_year)





#select only data from the united states

dfs = df[df['Country'] == 'United States']



#calculate average temp per year

dfa = pd.DataFrame()

years = dfs['year'].unique()

for i in years:

    df_avg = dfs[dfs['year'] == i]['AverageTemperature'].mean()

    df_new = (dfs[dfs['year'] == i]).head(1)

    df_new['AverageTemperature'] = df_avg

    dfa = dfa.append(df_new)
#drop and plot temps below 9 degrees

df_nine = dfa[dfa['AverageTemperature'] >= 9]

df_nine.plot.scatter(x='year', y='AverageTemperature', c = 'AverageTemperature', cmap ='coolwarm')
ss.head()
#drop unneaded columns

ss.drop(['Unnamed: 2', 'Unnamed: 3','Unnamed: 4','Unnamed: 5',

         'Unnamed: 6', 'Unnamed: 7','Unnamed: 8', 'Unnamed: 9'], axis=1, inplace=True)





#merge sun spot data and average temp data

dfsc = pd.merge(dfa, ss, on=['year'])
plt.scatter(x=ss['year'], y=ss['suns_spot_number'])

plt.ylabel('Sun Spot Number')

plt.xlabel('Years')

plt.title('Number of Sunspots since 1700')
cd.head()
#average CO2 ppm per year

dfc = pd.DataFrame()

years = cd['Year'].unique()

for i in years:

    df_avg = cd[cd['Year'] == i]['Carbon Dioxide (ppm)'].mean()

    df_new = (cd[cd['Year'] == i]).head(1)

    df_new['Carbon Dioxide (ppm)'] = df_avg

    dfc = dfc.append(df_new)

 

#change Year column to year

dfc.rename(index=str, columns={"Year": "year"}, inplace=True)



#merge CO2 data with temp and sun spot data

dfcss = pd.merge(dfsc, dfc, on=['year'])



#drop unwanted columns

dfcss.drop(['Seasonally Adjusted CO2 (ppm)', 

           'Carbon Dioxide Fit (ppm)', 

           'Seasonally Adjusted CO2 Fit (ppm)',

          'Decimal Date',

          'Month'], inplace=True, axis=1)
sns.lmplot(x='year', y='Carbon Dioxide (ppm)', data=dfcss)

sns.heatmap(dfcss.corr())
sns.lmplot(x='AverageTemperature', y='Carbon Dioxide (ppm)', data =dfcss)
sns.lmplot(x='AverageTemperature', y='suns_spot_number', data =dfcss)