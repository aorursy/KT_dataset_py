# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno



import bq_helper

from bq_helper import BigQueryHelper

import re

from fancyimpute import KNN

from pandas import DataFrame



# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def initiate_BigQuery():

    """This function will initiate the WDI Table for BigQuery



        Args:

            None



        Returns: None



        """

    wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="worldbank_wdi")



    #Initiate the BigQuery database assistant

    bq_assistant2 = BigQueryHelper("patents-public-data", "worldbank_wdi")

    print(bq_assistant2.list_tables())

    print(bq_assistant2.table_schema('wdi_2016'))
def call_query(indicators, max_querysize, bq_assistant, wdi):

    """This function searches and returns a list of list of our variables



        Args:

            indicators: This is a list of the WDI indicator codes of the desired variables

            max_querysize: This is a BigQuery parameter for specifing the max size of each query.

            bq_assistant: The BigQueryTool for quering

            wdi: The name of our database



        Returns: A list of list containing the variables we have searched for



        """

    query_list = indicators

    df_list = []



    for query_item in query_list:

        query = """

                  SELECT country_name, year, indicator_code, indicator_name, indicator_value

                  FROM `patents-public-data.worldbank_wdi.wdi_2016`

                  WHERE indicator_code LIKE '%OURQUERY%'

                  ORDER BY country_name, year 

                  """

        updated_query = query.replace("OURQUERY", query_item)

        print("Now running Query for item: ", query_item)

    

        query_size = bq_assistant.estimate_query_size(updated_query)

        print('The query size will be: ', query_size)

    

        query_response = wdi.query_to_pandas_safe(updated_query, max_gb_scanned=max_querysize)

        print(query_response.shape, "\n")

    

    

        file_name = query_response.indicator_name[0]

    

        for k in file_name.split("\n"):

            name = re.sub(r"[^a-zA-Z0-9]+", ' ', k)

            real_name = str(name) + str('.csv')

        

        

        query_response.to_csv(real_name)

        real_name = query_response

        df_list.append(real_name)

    

    print("Query Done. Please see output for individual files. \n")

    return df_list

def clean_query(my_list):

    """This function cleans up the list of list containing our data



        Args:

            my_list: our list of list conaing the big query result 



        Returns: a cleaned list, with correct column names 



        """

    clean_list = []

    for number, i in enumerate(my_list):

        column_name = i.indicator_name[0]

        column = i.rename(columns={'indicator_value':str(column_name)}, inplace=True)

        df = DataFrame(my_list[int(number)],columns=["country_name", "year", "indicator_code", "indicator_name", str(column_name)])

        df.drop(['indicator_code', 'indicator_name'], axis = 1, inplace=True)

        clean_list.append(df)

        

    print("Query has been Cleaned \n")

    return clean_list
def convert_to_df(cleaned_list):

    """This function converts our list of list to a dataframe



        Args:

            cleaned_list: the cleaned list returned from the clean_list function



        Returns: Panda dataframe containing all our data



        """

    final_df = cleaned_list[0]

    

    for i in cleaned_list[1:]:

        final_df = pd.concat([final_df, i.iloc[:,2:3]], axis=1)

    

    final_df.to_csv('Unparsed Data .csv',index=False)

    

    print("Convert to Dataframe: ", final_df.shape)

    print("Query Converted to a Dataframe \n")

    print("See output for full country list named UNPARSED DATA \n")

    return final_df
def parse_country(dataframe, country_list):

    """This function parses our data for our listed countries



        Args:

            dataframe: our cleaned dataframe

            country_list: A list containing the countries of the world or economic unions which we are interested in



        Returns: a dataframe with only our desired countries



        """

    country_df = dataframe[dataframe.country_name.str.contains('|'.join(country_list), na=False)]

    print("Parse Country: ", country_df.shape)

    return country_df
# my_countries = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra',

#        'Angola', 'Antigua and Barbuda', 'Argentina',

#        'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',

#        'Bahamas, The', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',

#        'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia',

#        'Bosnia and Herzegovina', 'Botswana', 'Brazil',

#        'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria',

#        'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon',

#        'Canada', 'Cayman Islands',

#        'Central African Republic',

#        'Chad', 'Channel Islands', 'Chile', 'China', 'Colombia', 'Comoros',

#        'Congo, Dem. Rep.', 'Congo, Rep.', 'Costa Rica', "Cote d'Ivoire",

#        'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic',

#        'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',

#        'Egypt, Arab Rep.', 'El Salvador', 'Equatorial Guinea', 'Eritrea',

#        'Estonia', 'Ethiopia', 'European Union',

#        'Faroe Islands', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',  'Germany',

#        'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guam',

#        'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',

#        'Honduras', 'Hong Kong SAR, China', 'Hungary', 

#        'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq',

#        'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan',

#        'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',

#        'Korea, Dem. People�s Rep.', 'Korea, Rep.', 'Kosovo', 'Kuwait',

#        'Kyrgyz Republic', 'Lao PDR', 'Latvia', 'Lebanon',

#        'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania',

#        'Luxembourg', 'Macao SAR, China', 'Macedonia, FYR', 'Madagascar',

#        'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',

#        'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro',

#        'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal',

#        'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua',

#        'Niger', 'Nigeria', 'Northern Mariana Islands',

#        'Norway', 'Oman', 'Pacific island small states', 'Pakistan',

#        'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru',

#        'Philippines', 'Poland', 'Portugal', 

# 'Puerto Rico', 'Qatar', 'Romania',

#        'Russian Federation', 'Rwanda', 'Samoa', 'San Marino',

#        'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',

#        'Seychelles', 'Sierra Leone', 'Singapore',

#        'Sint Maarten (Dutch part)', 'Slovak Republic', 'Slovenia',

#        'Small states', 'Solomon Islands', 'Somalia', 'South Africa',

#        'South Sudan', 'Spain',

#        'Sri Lanka', 'St. Kitts and Nevis', 'St. Lucia',

#        'St. Martin (French part)', 'St. Vincent and the Grenadines', 'Sudan', 'Suriname',

#        'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic',

#        'Tajikistan', 'Tanzania', 'Thailand', 'Togo',

#        'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',

#        'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'Uganda',

#        'Ukraine', 'United Arab Emirates', 'United Kingdom',

#        'United States', 'Uruguay', 'Uzbekistan',

#        'Vanuatu', 'Venezuela, RB', 'Vietnam', 'Virgin Islands (U.S.)',

#        'West Bank and Gaza', 'World', 'Yemen, Rep.', 'Zambia', 'Zimbabwe']



def fill_values(dataset, neighbors, country_list):

    """A K- Nearest Neighbor model which fills the NAs in our dataframe



        Args:

            dataset; The dataset

            neighbors: A KNN parameter for the neigbors used in modelling. Usually integer 3 or 5 

            country_list: A list containing our desired countries



        Returns: a dataframe with the possible NA's filled



        """

    dataset_columns = list(dataset.columns)

    dataset.replace(0, np.nan, inplace=True)



    filled_data = pd.DataFrame()

    for country in country_list:

        print(country)

        iteration_incomplete_dataset = dataset[dataset.isin([country]).any(axis=1)]

        iteration_incomplete_dataset.drop(['country_name'], axis=1, inplace=True)

        filled_country_data = pd.DataFrame(KNN(k=neighbors).fit_transform(iteration_incomplete_dataset))

        complete_data = filled_country_data

        complete_data.insert(0, 'Country', country)

        filled_data = pd.concat([filled_data, complete_data])

    

    filled_data.columns = dataset_columns

    filled_data.year = filled_data.year.astype(int)

    return filled_data
def save_data(dataframe):

    dataframe.to_csv('My_dataset.csv',index=False) #output our result
def create_data(indicators, country_list, bq_assistant, neighbors, fill_missing=True):

    """The final function to call all other functions



        Args:

            indicators: Our WDI indicator code variables

            country_list: A list of our desired countries

            neighbor: KNN nearest neighbor

            bq_assistant: The BigQuery Object to initialize our query

            fill_missing: A True or False variable to employ KNN to fill NAs



        Returns: a dataframe with the desired variables



        """

    initiate_BigQuery()

    

    raw_query = call_query(indicators, querysize, bq_assistant, wdi)

    

    clean_data = clean_query(raw_query)

    

    clean_dataframe = convert_to_df(clean_data)

    

    country_dataframe = parse_country(clean_dataframe, country_list)

    

    

    final = pd.DataFrame()

    if fill_missing == True:

        completed_data = fill_values(country_dataframe, neighbors, country_list)

        final = completed_data

        save_data(final)

    else:

        final = country_dataframe

        save_data(final)

    



    return final
#Please input a list for the WDI indicators which you are interested in as well as the countries you would like to spool the data for.

# It should be in a list format.



indicators_list = ['FR.INR.LEND', 'FR.INR.DPST', 'FM.LBL.BMNY.GD.ZS', 'SH.DYN.NMRT', 'SE.SEC.ENRL.GC.FE.ZS'] #list of WDI indicator code(s)

country_list = ["Nigeria", "Niger", 'France', "United Kingdom"] #list of countries

fill_NA = True #String indicating whether user wants to employ the KNN nearest neighbors models to fill NA values input TRUE or FALSE

querysize = 10 #A value in Gigabyte representing the maximum computer resource which each query will employ

neighbors = 5 #A value for the KNN model recommeded: Use 3 or 5





wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="worldbank_wdi")

bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")
data = create_data(indicators_list, country_list,bq_assistant, neighbors, fill_missing=fill_NA)

data.head(50)