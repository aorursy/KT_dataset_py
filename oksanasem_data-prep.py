# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', sep=',')

df['Date'] = pd.to_datetime(df['Date'])

train_last_date = df.Date.unique()[-1]

print(f"Dataset has training data untill : {train_last_date}")


wpop = pd.read_csv('/kaggle/input/worldpopulationbyage/WPP2019_PopulationByAgeSex_Medium.csv')



country_mapper = {

'Iran (Islamic Republic of)' : "Iran",

'Bolivia (Plurinational State of)' : 'Bolivia',

'Brunei Darussalam' : 'Brunei',

'Congo' : 'Congo (Kinshasa)',

'Democratic Republic of the Congo' : "Congo (Brazzaville)",

"Côte d'Ivoire": "Cote d'Ivoire",

"Gambia" : "Gambia, The",

"Republic of Korea": "Korea, South",

"Republic of Moldova": "Moldova",

'Réunion' : "Reunion",

'Russian Federation' : "Russia",

'China, Taiwan Province of China' : "Taiwan*",

"United Republic of Tanzania": "Tanzania",

"Bahamas": "The Bahamas",

"Gambia": "The Gambia",

"United States of America (and dependencies)" : "US",

"Venezuela (Bolivarian Republic of)" : "Venezuela",

'Viet Nam' : "Vietnam"}



def rename_countries(x, country_dict):

    new_name = country_dict.get(x)

    if new_name is not None:

        #print(x, "-->", new_name)

        return new_name

    else:

        return x



wpop = wpop[wpop['Time']==2020].reset_index(drop=True)

wpop['Location'] = wpop.Location.apply(lambda x : rename_countries(x, country_mapper))

clean_wpop = wpop[wpop['Location'].isin(df['Country_Region'].unique())].reset_index()



population_distribution = []

for country, gpdf in clean_wpop.groupby("Location"):

    aux = {f"age_{age_grp}": tot for age_grp, tot in zip(gpdf.AgeGrp, gpdf.PopTotal)}

    aux["Country_Region"] = country

    population_distribution.append(aux)

    

df_pop_distrib = pd.DataFrame(population_distribution)



# add missing countries with median values

no_data = []

for country in df['Country_Region'].unique():

    if country not in df_pop_distrib['Country_Region'].unique():

        aux = df_pop_distrib.drop('Country_Region', axis=1).median(axis=0).to_dict()

        aux["Country_Region"] = country

        no_data.append(aux)

df_no_data = pd.DataFrame(no_data)



df_pop_distrib = pd.concat([df_pop_distrib, df_no_data], axis=0)



# normalize features

norm_pop_distrib = df_pop_distrib.drop("Country_Region", axis=1).div(df_pop_distrib.drop("Country_Region", axis=1).sum(axis=1), axis=0)

norm_pop_distrib['total_pop'] = df_pop_distrib.drop("Country_Region", axis=1).sum(axis=1)

norm_pop_distrib["Country_Region"] = df_pop_distrib["Country_Region"]



del df_pop_distrib

del df_no_data

# del clean_wpop

# del wpop



df = df.merge(norm_pop_distrib, on="Country_Region", how='left')

df.shape
wpop.sample(10)
#https://ourworldindata.org/smoking#prevalence-of-smoking-across-the-world

smokers = pd.read_csv('/kaggle/input/smokingstats/share-of-adults-who-smoke.csv')

smokers = smokers[smokers.Year == 2016].reset_index(drop=True)



smokers_country_dict = {'North America' : "US",

 'Gambia' : "The Gambia",

 'Bahamas': "The Bahamas",

 "'South Korea'" : "Korea, South",

'Papua New Guinea' : "Guinea",

 "'Czech Republic'" : "Czechia",

 'Congo' : "Congo (Brazzaville)"}



smokers['Entity'] = smokers.Entity.apply(lambda x : rename_countries(x, smokers_country_dict))



no_datas_smoker = []

for country in df['Country_Region'].unique():

    if country not in smokers.Entity.unique():

        mean_score = smokers[['Smoking prevalence, total (ages 15+) (% of adults)']].mean().to_dict()

        mean_score['Entity'] = country

        no_datas_smoker.append(mean_score)

no_data_smoker_df = pd.DataFrame(no_datas_smoker)   

clean_smoke_data = pd.concat([smokers, no_data_smoker_df], axis=0)[['Entity','Smoking prevalence, total (ages 15+) (% of adults)']]

clean_smoke_data.rename(columns={"Entity": "Country_Region",

                                  "Smoking prevalence, total (ages 15+) (% of adults)" : "smokers_perc"}, inplace=True)



df = df.merge(clean_smoke_data, on="Country_Region", how='left')

df.shape
countries = list(df.Country_Region.unique())
healht_info = pd.read_csv('../input/health-nutrition-and-population-statistics/data.csv')

#healht_info.sample(5)



health_cols_2014 = [

'GNI per capita, Atlas method (current US$)',

       'Health expenditure per capita (current US$)',

       'Health expenditure per capita, PPP',

       'Health expenditure, private (% of GDP)',

       'Health expenditure, private (% of total health expenditure)',

       'Health expenditure, public (% of GDP)',

       'Health expenditure, public (% of government expenditure)',

       'Health expenditure, public (% of total health expenditure)',

       'Health expenditure, total (% of GDP)',

        'Prevalence of overweight (% of adults)']

health_cols_2015 = ['Diabetes prevalence (% of population ages 20 to 79)',]

health_BCG_col =['Immunization, BCG (% of one-year-old children)',]

health_cols_index = ['Country Name', 'Country Code', 'Indicator Name']



healht1 = healht_info[healht_info['Indicator Name'].isin(health_cols_2014)].pivot(index ='Country Code', columns ='Indicator Name', values = '2014').reset_index()

healht2 = healht_info[healht_info['Indicator Name'].isin(health_cols_2015)].pivot(index ='Country Code', columns ='Indicator Name', values = '2015').reset_index()

healht3 = healht_info[healht_info['Indicator Name'].isin(health_BCG_col)].pivot(index ='Country Code', columns ='Indicator Name', values = [ '1980', '1990', '2000'])

healht3.columns = healht3.columns.get_level_values(0)

healht3.columns = [' '.join(col).strip() for col in healht3.columns.values]

healht3 = healht3.add_prefix('BCG_')

healht3 = healht3.reset_index()

#healht1.drop(columns=['Indicator Name'], axis=1, inplace=True)



health_countries = healht_info[['Country Code','Country Name']].drop_duplicates(subset=['Country Code','Country Name'], keep="first", inplace=False)

#health_countries



healht_merged = health_countries.merge(healht1, on='Country Code').merge(healht2, on='Country Code').merge(healht3, on='Country Code')

healht_merged.info()
corruption_info = pd.read_csv('../input/corruption-index/index.csv')

corruption_info.sample(5)
merged1 = healht_merged.merge(corruption_info[['Country Code', 'Corruption Perceptions Index (CPI)']], on='Country Code', how='left')

#merged1.info()



country_mapper = {

'Iran (Islamic Republic of)' : "Iran",

'Bolivia (Plurinational State of)' : 'Bolivia',

'Brunei Darussalam' : 'Brunei',

    'The Bahamas': 'Bahamas',

'Congo' : 'Congo (Kinshasa)',

'Democratic Republic of the Congo' : "Congo (Brazzaville)",

"Côte d'Ivoire": "Cote d'Ivoire",

"Gambia" : "Gambia, The",

"Republic of Korea": "Korea, South",

"Republic of Moldova": "Moldova",

'Réunion' : "Reunion",

'Russian Federation' : "Russia",

'China, Taiwan Province of China' : "Taiwan*",

"United Republic of Tanzania": "Tanzania",

"Bahamas": "The Bahamas",

"Gambia": "The Gambia",

"United States" : "US",

"Venezuela, RB" : "Venezuela",

'Viet Nam' : "Vietnam",

'Egypt, Arab Rep.':'Egypt',

'Czech Republic': 'Czechia',

'Macedonia, FYR':'North Macedonia',

'Gambia, The':'Gambia',

'Iran, Islamic Rep.':'Iran',

'Slovak Republic':'Slovakia',

'Korea, Dem. People’s Rep.':'Korea, South',

'Kyrgyz Republic':'Kyrgyzstan',

    'Syrian Arab Republic':'Syria'}

def rename_countries(x, country_dict):

    new_name = country_dict.get(x)

    if new_name is not None:

        #print(x, "-->", new_name)

        return new_name

    else:

        return x





merged1['Country Name'] = merged1['Country Name'].apply(lambda x : rename_countries(x, country_mapper))





list(set(countries) - set(merged1.loc[:,'Country Name'].unique()))
merged1.info()
df.shape
df = df.merge(merged1, left_on="Country_Region",right_on="Country Name", how='left')

df.drop(columns=['Country Code', 'Country Name'], axis=1, inplace=True)

df.info()
def concat_country_province(country, province):

    if not isinstance(province, str):

        return country

    else:

        return country+"_"+province



# Concatenate region and province for training

df["Country_Region"] = df[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)

country_info = pd.read_csv('/kaggle/input/countryinfo/covid19countryinfo.csv')

country_info = country_info[~country_info.country.isnull()].reset_index(drop=True)

country_info.drop([ c for c in country_info.columns if c.startswith("Unnamed")], axis=1, inplace=True)

country_info.drop(columns=['pop', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64', 'sex65plus', 'medianage', "smokers", "sexratio"],

                  axis=1,

                  inplace=True)

##

country_info = country_info.drop(country_info[country_info.country=='Mali'].index)

#country_info.loc[country_info.country=='Mali','nonessential':] = pd.to_datetime('2020-03-17')

#country_info.loc[country_info.country=='Mali','gathering':] = pd.to_datetime('2020-03-17')

#country_info.loc[country_info.country=='Mali','gatheringlimit':] = 50

####

# Columns with dates

country_info["quarantine"] = pd.to_datetime(country_info["quarantine"])

country_info["publicplace"] = pd.to_datetime(country_info["publicplace"])

country_info["gathering"] = pd.to_datetime(country_info["gathering"])

country_info["nonessential"] = pd.to_datetime(country_info["nonessential"])

country_info["schools"] = pd.to_datetime(country_info["schools"])

country_info["firstcase"] = pd.to_datetime(country_info["firstcase"])

##

country_info['gdp2019'] = country_info['gdp2019'].str.replace(',', '')

country_info['healthexp'] = country_info['healthexp'].str.replace(',', '')









same_state = []

for country in df["Province_State"].unique():

    if country in country_info.country.unique():

        same_state.append(country)

    else:

        pass

        # This part can help matching different external dataset and find corresponding countries

        #print(country)

        #matches = []

        #scores = []

        #if str(country)=="nan":

        #    continue

        #for possible_match in country_info.country.unique():

        #    matches.append(possible_match)

        #    scores.append(fuzz.partial_ratio(country, possible_match))

            

        #top_5_index = np.argsort(scores)[::-1][:5]

        #print(np.array(matches)[top_5_index])

        #print(np.array(scores)[top_5_index])

        #print("-------------------")

        

country_to_state_country = {}

for state in same_state:

    #print(state)

    #print(df[df["Province/State"]==state]["Country/Region"].unique())

    #print("----")

    country_to_state_country[state] = df[df["Province_State"]==state]["Country_Region"].unique()[0]+"_"+state



country_info['country'] =country_info[["country", "region"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)                                                                      





dates_info = ["publicplace", "gathering", "nonessential", "quarantine", "schools","firstcase"]

coutry_merge_info = country_info[["country", "density", "urbanpop", "hospibed", "lung",

                                  "femalelung", "malelung",'gdp2019', 'healthexp', 'healthperpop', 'fertility'] + dates_info]



cols_median = ["density", "urbanpop", "hospibed", "lung", "femalelung", "malelung",'gdp2019', 'healthexp', 'healthperpop', 'fertility']

coutry_merge_info.loc[:, cols_median] = coutry_merge_info.loc[:, cols_median].apply(lambda x: x.fillna(x.median()),axis=0)





merged = df.merge(coutry_merge_info, left_on="Country_Region", right_on="country", how="left")

merged.loc[:, cols_median] = merged.loc[:, cols_median].apply(lambda x: x.fillna(x.median()),axis=0)



country_dates_info = country_info[["country", "publicplace", "gathering", "nonessential", "quarantine", "schools","firstcase"]]







def dates_diff_days(date_curr, date_):

    if date_curr>date_:

        return (date_curr - date_).days

    else :

        return 0





for col in dates_info:

    #print(merged.shape)

    merged[col+'_days'] =merged[["Date", col]].apply(lambda x : dates_diff_days(x[0], x[1]), axis=1)                                                                      



print(merged.shape)

#drop_country_cols = [x for x in merged.columns if x.startswith("country")] + dates_info

drop_country_cols = [x for x in merged.columns if x.startswith("country")]

merged.drop(columns=drop_country_cols, axis=1, inplace=True)

print(merged.shape)
merged.Country_Region.value_counts().mean()
merged.sample(10)
weather_info = pd.read_csv('../input/weather-info/training_data_with_weather_info_week_2.csv')

weather_info.sample(5)
weather_info.Date.min(), weather_info.Date.max() , weather_info.shape
merged.Date.min(), merged.Date.max(), merged.shape
weather_info["Country_Region"] = weather_info[["Country_Region", "Province_State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)

weather_info["Date"] = pd.to_datetime(weather_info["Date"])



merged_ = merged.merge(weather_info[['temp','min','max','stp','wdsp','prcp','fog','Country_Region', 'Date']], on=["Country_Region", 'Date'])

merged_.shape
merged_.Country_Region.value_counts().mean()
hh_info = pd.read_csv('../input/covid19aug/augmented/train_aug.csv')

hh_cols = ['hh%1',

'hh%2-3',

'hh%4-5',

'hh%6+',

'hh65+']

hh_info.sample(5)
def concat_country_province_(country, province):

    if not isinstance(province, str):

        return country

    else:

        if country==province:

            return country

        else:

            return country+"_"+province



# Concatenate region and province for training

hh_info["Country_Region"] = hh_info[["Province/State", "Country/Region"]].apply(lambda x : concat_country_province_(x[0], x[1]), axis=1)

hh_info.loc[hh_info["Country/Region"]=='US', 'hh%1'] = 28

hh_info.loc[hh_info["Country/Region"]=='US', 'hh%2-3'] = 49

hh_info.loc[hh_info["Country/Region"]=='US', 'hh%4-5'] = 19

hh_info.loc[hh_info["Country/Region"]=='US', 'hh%6+'] = 4

hh_info.loc[hh_info["Country/Region"]=='US', 'hh65+'] = 28
hh_info_ = hh_info[['Country_Region']+hh_cols].drop_duplicates(subset=['Country_Region'], keep="first", inplace=False)
hh_info_.info()
#merged_ = merged_.merge(hh_info_, on=["Country_Region"], how='left')

#merged_.shape
merged_.info()
merged_.to_csv('enriched_covid_19_week_2_2.csv', index=None)