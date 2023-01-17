#all the imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#grab data here
#covid data
covid_master_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

#world-happiness-index
world_happiness_df = pd.read_csv("../input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv")

#here we will do all the group by, aggregation etc.


########doing a group by on Country_Region
covid_master_df_COUNTRY = covid_master_df.groupby(['Country_Region'], as_index=False).max()
########adding a column that lists the fatalities to confirmed cases ratio
covid_master_df_COUNTRY['confirmed_to_fatal_ratio'] = covid_master_df_COUNTRY['Fatalities']/covid_master_df_COUNTRY['ConfirmedCases']
covid_master_df_COUNTRY = covid_master_df_COUNTRY.sort_values(['ConfirmedCases'], ascending=False)

#Fixing some same but differently typed values
covid_master_df_COUNTRY["Country_Region"] = covid_master_df_COUNTRY["Country_Region"].replace("US", "United States")

covid_master_df_COUNTRY.head()


#########merge/join with world_happiness_df data

world_happiness_df = world_happiness_df.rename(columns={"Country name":"Country_Region"})

covid_master_df_COUNTRY_with_world_happiness = pd.merge(covid_master_df_COUNTRY, world_happiness_df, how='inner', on=["Country_Region"])
covid_master_df_COUNTRY_with_world_happiness.head()


#Do the plotting here
#bubble plot of top 10 affected countries
plt.figure(figsize=(15,10))
plt.scatter(covid_master_df_COUNTRY_with_world_happiness['Country_Region'].head(10), 
            covid_master_df_COUNTRY_with_world_happiness['ConfirmedCases'].head(10), 
            s=np.reciprocal(covid_master_df_COUNTRY_with_world_happiness['ConfirmedCases'].head(10).max()/covid_master_df_COUNTRY_with_world_happiness['ConfirmedCases'].head(10))*10000, 
            c=covid_master_df_COUNTRY_with_world_happiness['confirmed_to_fatal_ratio'].head(10), cmap="Reds", alpha=0.5, edgecolors="grey", linewidth=2)

plt.bar(covid_master_df_COUNTRY_with_world_happiness['Country_Region'].head(10),covid_master_df_COUNTRY_with_world_happiness['confirmed_to_fatal_ratio'].head(10)*1000000, alpha=0.4)

plt.bar(covid_master_df_COUNTRY_with_world_happiness['Country_Region'].head(10),covid_master_df_COUNTRY_with_world_happiness['Ladder score'].head(10)*10000, alpha=0.4, width=0.5)


# plt.grid(True, which='both')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.xticks(rotation='vertical',fontsize='30')
plt.yticks(covid_master_df_COUNTRY['ConfirmedCases'].head(10), fontsize='15')
plt.title("COVID19 Global Forecasting: Top 10 Countries by confirmed cases and the mortality ratio")
plt.show()

#plotting specifically happiness_index with fatality ratio
covid_master_df_COUNTRY_with_world_happiness_sorted = covid_master_df_COUNTRY_with_world_happiness.sort_values(['confirmed_to_fatal_ratio'], ascending=False)
covid_master_df_COUNTRY_with_world_happiness_sorted['Ladder score']


#scatter plot of Ladder score against confirmed_to_fatal_ratio
plt.figure(figsize=(60,30))
plt.scatter((covid_master_df_COUNTRY_with_world_happiness_sorted['Ladder score']/covid_master_df_COUNTRY_with_world_happiness_sorted['Ladder score'].max())/4,
            covid_master_df_COUNTRY_with_world_happiness_sorted['confirmed_to_fatal_ratio'],             
            s = 100)

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.xlabel('Ladder Score')
plt.ylabel('confirmed_to_fatal_ratio')
plt.yticks(rotation='vertical',fontsize='30')
plt.xticks(covid_master_df_COUNTRY['confirmed_to_fatal_ratio'], fontsize='15', rotation='vertical')
plt.title("COVID19 Global Forecasting: all countries Ladder Score(Happiness Index) to Confirmed_to_fatal_ratio")
plt.show()