import pandas as pd
filename = "/kaggle/input/iso-country-codes-with-alternative-country-names/country_iso_codes_expanded.csv"

country_iso_codes_expanded = pd.read_csv(filename)



filename = "/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv"

df_to_standardize = pd.read_csv(filename)

country_column = "Country/Region"



df_to_standardize["country_standardized"], df_to_standardize["country_iso"] = ["", ""]

old_country_names = df_to_standardize[country_column].unique()



for old_country in old_country_names:

    if old_country not in country_iso_codes_expanded['country'].unique():

        column_index = 0 

        for column_index in range(len(country_iso_codes_expanded.columns[5:])):

            alternative_country_column = country_iso_codes_expanded.loc[:,"alternative_country_name_" + str(column_index)]

            matching_result = old_country == alternative_country_column

            match_index = [i for i, x in enumerate(matching_result) if x]

            if match_index:

                standardized_country = country_iso_codes_expanded.iloc[match_index[0], 0] 

                country_iso = country_iso_codes_expanded.iloc[match_index[0], 4]

                df_to_standardize.loc[df_to_standardize[country_column] == old_country, 

                                      "country_standardized"] = standardized_country 

                df_to_standardize.loc[df_to_standardize[country_column] == old_country, 

                                      "country_iso"] = country_iso

                break

            column_index += 1

    else:

        df_to_standardize.loc[df_to_standardize[country_column] == old_country, "country_standardized"] = old_country

        country_iso = country_iso_codes_expanded.loc[country_iso_codes_expanded['country'] == old_country, "ISO 3166-2"].item()

        df_to_standardize.loc[df_to_standardize[country_column] == old_country, "country_iso"] = country_iso  
df_to_standardize.head()