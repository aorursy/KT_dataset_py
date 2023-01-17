import pandas as pd

import warnings
# Global vars 

csv_containment_measures_data_location = "../input/COVID 19 Containment measures data.csv"



countries=['Austria','Germany','United Kingdom','Vietnam','South Korea','Singapore','Israel','Japan','Sweden','San Marino','Slovenia','Canada','Hong Kong','Taiwan','United States','European Union','Thailand','Italy','Czechia','Australia','Trinidad and Tobago','Qatar','North Korea','New Zealand','Colombia','Romania','France','Portugal','Spain','Belgium','Luxembourg','Albania','Andorra','Azerbaijan','Belarus','Bosnia and Herzegovina','Bulgaria','Denmark','Estonia','Cyprus','Croatia','Finland','Georgia','Hungary','Latvia','Lithuania','Greece','Moldova','Malta','Monaco','Netherlands','Iceland','Guernsey','Macedonia','Ireland','Vatican City','Jersey','Kosovo','Kazakhstan','Poland','Turkey','Ukraine','Slovakia','Serbia','Switzerland','Norway','Montenegro','Iran','Liechtenstein','Russia','Mexico','Egypt','Palestine','Malaysia','Nepal','Afghanistan','Iraq','Faroe Islands','Philippines','Kuwait','South Africa','Armenia','Pakistan','Brazil','Costa Rica','Panama','India','Bahrain','United Arab Emirates','Kyrgyzstan','Indonesia','Namibia','Morocco','Uganda']



# Remove warnings when filtering

warnings.filterwarnings("ignore", 'This pattern has match groups')
"""

Read CSV file from disk, select fields, change column names, change column types and filter countries

"""



# print multi-line cells

pd.set_option('display.max_colwidth', -1)



df = pd.read_csv(csv_containment_measures_data_location)

df = df[["Country","Date Start","Description of measure implemented", "Keywords"]]

df = df.rename(columns={"Description of measure implemented": "Description", "Date Start": "StartDate"})

df["StartDate"] = pd.to_datetime(df["StartDate"], format='%b %d, %Y')



df = df[df["Country"].isin(countries)]

df = df.sort_values(by=['StartDate'])

df


containment_measures = [

    {

        "label":"international_travel", 

        "onlyFirst":True, 

        "filter":"(international travel|travel ban)",

        "severity":1

    },{

        "label":"bussiness_closure",

        "filter":"general nonessential business suspension",

        "onlyFirst":True,

        "severity":2

    },{

        "label":"isolation",

        "filter":"cluster isolation|compulsory isolation",

        "onlyFirst":True,

        "severity":3

    },{

        "label":"state_emergency",

        "filter":"state.*emergency",

        "onlyFirst":True,

        "severity":2

    },{

        "label":"cancellation",

        "filter":"cancellation",

        "onlyFirst":True,

        "severity":1

    },{

        "label":"edu_closure",

        "filter":"(school|university)*clo",

        "onlyFirst":True,

        "severity":2

    }

]
# Create empty dataframe 

df_filtered = df[df["Keywords"] == "qwerty"]



# Filter for each country

for country in countries:

    df_c = df[df["Country"] == country]

    for containment_measure in containment_measures:

        df_tmp = df_c[df_c["Keywords"].str.contains(containment_measure['filter'], na=False)]

        df_tmp = df_tmp.assign(Label=lambda x: containment_measure["label"])

        df_tmp = df_tmp.assign(Severity=lambda x: containment_measure["severity"])

        if len(df_tmp) == 0:

            continue

        if "onlyFirst" in containment_measure and containment_measure["onlyFirst"] == True:

            df_filtered = df_filtered.append(df_tmp.iloc[0], ignore_index=True)

        else: 

            df_filtered = df_filtered.append(df_tmp, ignore_index=True)



df_filtered['StartDate'].astype('datetime64[ns]')

df_filtered = df_filtered.sort_values(by=['Label'])

df_filtered
# df_filtered.to_csv("data/export_containment_measures.csv")