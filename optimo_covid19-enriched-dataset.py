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
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', sep=',')

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

clean_wpop = wpop[wpop['Location'].isin(df['Country/Region'].unique())].reset_index()



population_distribution = []

for country, gpdf in clean_wpop.groupby("Location"):

    aux = {f"age_{age_grp}": tot for age_grp, tot in zip(gpdf.AgeGrp, gpdf.PopTotal)}

    aux["Country/Region"] = country

    population_distribution.append(aux)

    

df_pop_distrib = pd.DataFrame(population_distribution)



# add missing countries with median values

no_data = []

for country in df['Country/Region'].unique():

    if country not in df_pop_distrib['Country/Region'].unique():

        aux = df_pop_distrib.drop('Country/Region', axis=1).median(axis=0).to_dict()

        aux["Country/Region"] = country

        no_data.append(aux)

df_no_data = pd.DataFrame(no_data)



df_pop_distrib = pd.concat([df_pop_distrib, df_no_data], axis=0)



# normalize features

norm_pop_distrib = df_pop_distrib.drop("Country/Region", axis=1).div(df_pop_distrib.drop("Country/Region", axis=1).sum(axis=1), axis=0)

norm_pop_distrib['total_pop'] = df_pop_distrib.drop("Country/Region", axis=1).sum(axis=1)

norm_pop_distrib["Country/Region"] = df_pop_distrib["Country/Region"]



del df_pop_distrib

del df_no_data

del clean_wpop

del wpop



df = df.merge(norm_pop_distrib, on="Country/Region", how='left')
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

for country in df['Country/Region'].unique():

    if country not in smokers.Entity.unique():

        mean_score = smokers[['Smoking prevalence, total (ages 15+) (% of adults)']].mean().to_dict()

        mean_score['Entity'] = country

        no_datas_smoker.append(mean_score)

no_data_smoker_df = pd.DataFrame(no_datas_smoker)   

clean_smoke_data = pd.concat([smokers, no_data_smoker_df], axis=0)[['Entity','Smoking prevalence, total (ages 15+) (% of adults)']]

clean_smoke_data.rename(columns={"Entity": "Country/Region",

                                  "Smoking prevalence, total (ages 15+) (% of adults)" : "smokers_perc"}, inplace=True)



df = df.merge(clean_smoke_data, on="Country/Region", how='left')

def concat_country_province(country, province):

    if not isinstance(province, str):

        return country

    else:

        return country+"_"+province



# Concatenate region and province for training

df["Country/Region"] = df[["Country/Region", "Province/State"]].apply(lambda x : concat_country_province(x[0], x[1]), axis=1)

# https://www.kaggle.com/koryto/countryinfo



country_info = pd.read_csv('/kaggle/input/countryinfo/covid19countryinfo.csv')

country_info = country_info[~country_info.country.isnull()].reset_index(drop=True)

country_info.drop([ c for c in country_info.columns if c.startswith("Unnamed")], axis=1, inplace=True)

country_info.drop(columns=['pop', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64', 'sex65plus', 'medianage', "smokers", "sexratio"],

                  axis=1,

                  inplace=True)

# Columns with dates

country_info["quarantine"] = pd.to_datetime(country_info["quarantine"])

country_info["restrictions"] = pd.to_datetime(country_info["restrictions"])

country_info["schools"] = pd.to_datetime(country_info["schools"])



same_state = []

for country in df["Province/State"].unique():

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

    country_to_state_country[state] = df[df["Province/State"]==state]["Country/Region"].unique()[0]+"_"+state



country_info['country'] = country_info.country.apply(lambda x : rename_countries(x, country_to_state_country))



coutry_merge_info = country_info[["country", "density", "urbanpop", "hospibed", "lung", "femalelung", "malelung"]]



cols_median = ["density", "urbanpop", "hospibed", "lung", "femalelung", "malelung"]

coutry_merge_info.loc[:, cols_median] = coutry_merge_info.loc[:, cols_median].apply(lambda x: x.fillna(x.median()),axis=0)





merged = df.merge(coutry_merge_info, left_on="Country/Region", right_on="country", how="left")

merged.loc[:, cols_median] = merged.loc[:, cols_median].apply(lambda x: x.fillna(x.median()),axis=0)



country_dates_info = country_info[["country", "restrictions", "quarantine", "schools"]]



def update_dates(a_df, col_update):

    """

    This creates a boolean time series with one after the start of confinements (different types : schools, restrictions or quarantine)

    """

    gpdf = a_df.groupby("Country/Region")

    new_col = gpdf.apply(lambda df : df[col_update].notnull().cumsum()).reset_index(drop=True)

    a_df[col_update] = new_col





for col in ["restrictions", "quarantine", "schools"]:

    print(merged.shape)

    merged = merged.merge(country_dates_info[["country", col]],

                          left_on=["Country/Region", "Date"],

                          right_on=["country", col],

                          how="left",

                          )

    update_dates(merged, col)



drop_country_cols = [x for x in merged.columns if x.startswith("country_")]

merged.drop(columns=drop_country_cols, axis=1, inplace=True)
merged.to_csv('enriched_covid_19.csv', index=None)