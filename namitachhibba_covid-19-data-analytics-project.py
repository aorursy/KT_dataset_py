import pandas as pd

import numpy as np

from datetime import datetime, timedelta

import matplotlib.pyplot as plt 
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

%matplotlib inline 
# import os

# os.chdir('/kaggle/working')

# os.getcwd()

# print(os.listdir("../input"))
# Importing  Confirmed Cases Dataset

# url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

# df_confirmed = pd.read_csv(url_confirmed, index_col="Country/Region")

df_confirmed = pd.read_csv("../input/time-series-data-covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",

                          header = 0, index_col="Country/Region")

df_confirmed.drop(['Lat', 'Long' ], axis=1, inplace=True)

df_confirmed.head(3)
# Country grouping on original dataframe        

gr_confirmed = df_confirmed.groupby("Country/Region").sum()

gr_confirmed.head(3)
# Checking for the missing values in the Confirmed Cases dataset



gr_confirmed.isnull().sum().sum()

gr_confirmed.isna().sum().sum()



# Hence, there are no missing values in our Confirmed Cases data. 
# Adding data for China (1-Jan to 21-Jan from a China CDC publication)

lab=[]

for i in range(1,22):

    lab.append("1/" + str(i) + "/20")

    gr_confirmed.insert(loc=i-1,column=lab[i-1], value=0)

gr_confirmed.loc["China"][0:10] = 20

gr_confirmed.loc["China"][10:21] = 310



# Remove Diamond princess and MS Zaandam

gr_confirmed = gr_confirmed.drop(["Diamond Princess", "MS Zaandam"])



gr_confirmed1= gr_confirmed.copy() # To have a copy of Confirmed Cases dataset in date format because ahead we are going to transform the data to days format

gr_confirmed1 = gr_confirmed1.reset_index()

# gr_confirmed.head(3)



# Dates are converted into no of days since 1/1/20 so that 1/1/20 corresponds to day 1

gr_confirmed_melt = gr_confirmed

dates = gr_confirmed_melt.keys()

FMT = '%m/%d/%y'



days = dates.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("1/1/20", FMT)).days)



days = days + 1 # +1 is to start the days from 1 instead of 0



gr_confirmed_melt.columns = days # = dates will convert the columns to date formats again



x_lag = gr_confirmed_melt.ge(10).idxmax(axis=1) # x_lag gives position of first value in each row which is greater than or equal to 10



for i in range(gr_confirmed_melt.shape[0]): # gr_confirmed_melt.shape[0] = 187 (no. of rows) and gr_confirmed_melt.shape[1] = 138 (no. of columns)

    gr_confirmed_melt.iloc[i] = gr_confirmed_melt.iloc[i].shift(periods=-x_lag[i]+1) # all data shift to one starting point 

# gr_confirmed_melt.head(3)



# Melting our Confirmed Cases dataset



gr_confirmed_melt = gr_confirmed_melt.reset_index()

gr_confirmed_melt = pd.melt(gr_confirmed_melt,id_vars= "Country/Region", 

                       value_vars=days, var_name="Days", 

                       value_name="Cumulative Confirmed Count").sort_values(["Country/Region","Days"], ignore_index=True)



# gr_confirmed_melt.set_index("Country/Region", inplace = True)

# gr_confirmed_melt.shape

gr_confirmed_melt.head(5)
# Importing Deaths Dataset

# url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

# df_death = pd.read_csv(url_death, index_col="Country/Region")

df_death = pd.read_csv('../input/time-series-data-covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv', 

                       header = 0, index_col="Country/Region")

df_death.drop(['Lat', 'Long'], axis=1, inplace=True)

df_death.head(3)
# Country grouping on original dataframe        

gr_death = df_death.groupby("Country/Region").sum()

gr_death.head(3)



# # Now adding Australian dataframe

# gr_death = pd.concat([gr_death, Aust_death])



# Adding data for China (1-Jan to 20-Jan) 

lab=[]

for i in range(1,22):

    lab.append("1/" + str(i) + "/20")

    gr_death.insert(i-1,lab[i-1],0)

gr_death.loc["China"][0:10] = 1

gr_death.loc["China"][10:21] = 1



# Remove Diamond princess

gr_death = gr_death.drop(["Diamond Princess", "MS Zaandam"])

# gr_death.to_csv("C:/Users/user/Downloads/COVID-19 Related Info/Deaths.csv")



gr_death1 = gr_death.copy() # To have a copy of death dataset in date format because ahead we are going to transform the data to days format

gr_death1 = gr_death1.reset_index()

# gr_death1.to_csv("C:/Users/user/Downloads/COVID-19 Related Info/Created Files/Deaths_datewise.csv")



gr_death.head(3)
# Checking for the missing values in the death Cases dataset



gr_death.isnull().sum().sum()

gr_death.isna().sum().sum()



# Hence, there are no missing values in our death Cases data. 
# Dates are converted into no of days since 1/1/20 so that 1/1/20 corresponds to day 1

gr_death_melt = gr_death

dates = gr_death_melt.keys()

FMT = '%m/%d/%y'



days = dates.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("1/1/20", FMT)).days)

days = days + 1

# +1 is to start the days from 1 instead of 0



gr_death_melt.columns = days # = dates will convert the columns to date formats again



x_lag  # x_lag gives position of first value in each row which is greater than or equal to 5 in confirmed cases dataset



for i in range(gr_death_melt.shape[0]): # gr_death_melt.shape[0] = 187 (no. of rows) and gr_death_melt.shape[1] = 138 (no. of columns)

    gr_death_melt.iloc[i] = gr_death_melt.iloc[i].shift(periods=-x_lag[i]+1) # all data shift to one starting point 

# gr_death_melt.head(3)



# Melting our Confirmed Cases dataset



gr_death_melt = gr_death_melt.reset_index()

gr_death_melt = pd.melt(gr_death_melt,id_vars= "Country/Region", 

                       value_vars=days, var_name="Days", 

                       value_name="Cumulative Death Count").sort_values(["Country/Region","Days"], ignore_index=True)



# gr_death_melt.set_index("Country/Region", inplace = True)

# gr_death_melt.to_csv("C:/Users/user/Downloads/COVID-19 Related Info/Created Files/Deaths_daywise_melted.csv")

# gr_death_melt.shape

gr_death_melt.head(5)
# Importing dataset for Recovered Cases

# url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

# df_recovered = pd.read_csv(url_recovered, index_col="Country/Region")

df_recovered = pd.read_csv("../input/time-series-data-covid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",

                           header = 0, index_col="Country/Region")



df_recovered.drop(['Lat', 'Long'], axis=1, inplace=True)

df_recovered.head(3)

# Country grouping on original dataframe        

gr_recovered = df_recovered.groupby("Country/Region").sum()



# Now adding Australian dataframe

# gr_death = pd.concat([gr_death, Aust_death])



# Adding data for China (1-Jan to 20-Jan) 

lab=[]

for i in range(1,22):

    lab.append("1/" + str(i) + "/20")

    gr_recovered.insert(i-1,lab[i-1],0)

gr_recovered.loc["China"][0:10] = 0

gr_recovered.loc["China"][10:21] = 0



# Remove Diamond princess

gr_recovered = gr_recovered.drop(["Diamond Princess", "MS Zaandam"])



# Adding the prefix to  all columns of Deaths dataset columns, which are dates to distingish them 

# from dates under Confirmed Cases dataset columns which are also same dates.

# gr_recovered = gr_recovered.add_prefix("Recovered Cases on ")



gr_recovered1 = gr_recovered.copy()  # To have a copy of Recovered Cases dataset in date format because ahead we are going to transform the data to days format

gr_recovered1 = gr_recovered1.reset_index()

# gr_recovered1.to_csv("C:/Users/user/Downloads/COVID-19 Related Info/Created Files/Recovered_datewise.csv")



# gr_recovered.shape

gr_recovered.head(3)
# Checking for the missing values in the Recovered Cases dataset



gr_recovered.isnull().sum().sum()

gr_recovered.isna().sum().sum()



# Hence, there are no missing values in our Recovered Cases data. 
# Dates are converted into no of days since 1/1/20 so that 1/1/20 corresponds to day 1

gr_recovered_melt = gr_recovered

dates = gr_recovered_melt.keys()

FMT = '%m/%d/%y'



days = dates.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("1/1/20", FMT)).days)



days = days + 1

# +1 is to start the days from 1 instead of 0



gr_recovered_melt.columns = days # = dates will convert the columns to date formats again



x_lag # x_lag gives position of first value in each row which is greater than or equal to 5 in confirmed cases data



for i in range(gr_recovered_melt.shape[0]): # gr_recovered_melt.shape[0] = 187 (no. of rows) and gr_recovered_melt.shape[1] = 138 (no. of columns)

    gr_recovered_melt.iloc[i] = gr_recovered_melt.iloc[i].shift(periods=-x_lag[i]+1) # all data shift to one starting point 

gr_recovered_melt.head(3)



#### Melting our Confirmed Cases dataset



gr_recovered_melt = gr_recovered_melt.reset_index()

gr_recovered_melt = pd.melt(gr_recovered_melt,id_vars= "Country/Region", 

                       value_vars=days, var_name="Days", 

                       value_name="Cumulative Recovered Count").sort_values(["Country/Region","Days"], ignore_index=True)



# gr_recovered_melt.set_index("Country/Region", inplace = True)

# gr_recovered_melt.to_csv("C:/Users/user/Downloads/COVID-19 Related Info/Created Files/Recovered_daywise_melted.csv")

# gr_recovered_melt.shape

gr_recovered_melt.head(5)
# url_test= 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv'

# df_test = pd.read_csv(url_test)

df_test = pd.read_csv("../input/testing-data-covid19/public/data/testing/covid-testing-all-observations.csv", header = 0)

df_test.head(2)
df_test.shape
df_test.columns
# Changing the "Date" column from object type to type datetime

df_test['Date'] = df_test['Date'].astype('datetime64[ns]')    



# Changing the format of the "Date" column to the one matching the dates in Confirmed Cases/Deaths and Recovered Cases datasets.

df_test['Date'] = df_test['Date'].dt.strftime('%m/%d/%y')

df_test.dtypes
df_test.drop(['ISO code','Source URL', 'Source label', 'Notes',

#               'Cumulative total',

#               'Daily change in cumulative total',

              'Cumulative total per thousand',

              'Daily change in cumulative total per thousand'], axis=1, inplace=True)

#               '7-day smoothed daily change',

#               7-day smoothed daily change per thousand,], 
df_test = df_test.drop(df_test[df_test["Entity"].isin(['India - people tested',

#               'France - tests performed',

              'France - people tested',

              'Italy - people tested',

              'Japan - tests performed',

              'Poland - people tested',

              'Singapore - people tested',

#               'Sweden - people tested',

              'Sweden - samples tested',                           

              'United States - tests performed (CDC) (incl. non-PCR)'

              ])].index)
df_test.head(5)
# Modifying the name of the "Entity" column to "Country/Region" as in confirmed and deaths datasets to join them using same 

# column with same name

# And renaming other columns to be more informative

df_test= df_test.rename(columns={"Entity":"Country/Region","Cumulative total": "Cumulative Testing Count",

                                "Daily change in cumulative total": "Daily Testing Count",

                                "7-day smoothed daily change": "7-day smoothed daily change in testing",

                                "7-day smoothed daily change per thousand":"7-day smoothed daily change per thousand in testing"

                                }) 

df_test.head(3)
# b) To get the country name from the value under Entity column like: "Argentina" from "Argentina - tests performed"

df_test["Country/Region"]= df_test["Country/Region"].str.split(" -", n=1, expand=True)

df_test.head(3)
df_test1 = df_test[["Country/Region", "Date", "Cumulative Testing Count"]]

df_test2 = df_test[["Country/Region", "Date", "Daily Testing Count"]]

df_test3 = df_test[["Country/Region", "Date", "7-day smoothed daily change in testing"]]

df_test4 = df_test[["Country/Region", "Date", "7-day smoothed daily change per thousand in testing"]]

data = [df_test1, df_test2, df_test3, df_test4]
index = ["Cumulative Testing Count", "Daily Testing Count", "7-day smoothed daily change in testing", "7-day smoothed daily change per thousand in testing"]

df_unmelted = []

for i, df in enumerate(data):

    df_test_unmelted = []

    df_test_unmelted = df.pivot_table(index="Country/Region", columns='Date')

    df_test_unmelted = df_test_unmelted[index[i]].reset_index()

    df_test_unmelted.columns.name = None

    df_unmelted.append(df_test_unmelted)
df_unmelted[0].head()
for index in range(2,4):

    lab=[]

    for i in range(1,8):

        df_unmelted[index] = df_unmelted[index].set_index("Country/Region")

        lab.append("01/0" + str(i) + "/20")

        df_unmelted[index].insert(loc=i-1,column=lab[i-1], value=np.nan)

        df_unmelted[index] = df_unmelted[index].reset_index()

        

df_unmelted[3].head()    
# Comparing the Confirmed Cases (or/Deaths/ Recovered Cases) dataset with Testing data for different country names for the same

# country or different countries in the two datasets using the gr_confirmed1 file - the datewise dataset copy of confirmed cases

# and the unmelted testing dataset (df_test_unmelted)



countries_in_either_datasets = pd.merge(gr_confirmed1["Country/Region"], df_unmelted[0],  how='outer',on= "Country/Region", indicator = True)

# countries_in_either_datasets



countries_only_in_df_ConfirmedData =countries_in_either_datasets[countries_in_either_datasets['_merge'] == 'left_only']

countries_only_in_df_TestData = countries_in_either_datasets[countries_in_either_datasets['_merge'] == 'right_only']



countries_only_in_df_TestData

# countries_only_in_df_ConfirmedData
# Doing this check we get to know that in gr_death and gr_confirmed and gr_recovered has 186 rows each and all values are common

# but in test_df there are just 82 rows out of which following rows are just in test_df and some values out of these are different 

# names for the same country presnt in the other 3 dataframes. So we rename those values and keep other which are not present

# in other 3 dataframes just like that, with NAs for their columns for those countries.



# gr_confirmed1 = gr_confirmed1.rename(index={'Taiwan*': 'Taiwan'})

# df_test_unmelted = df_test_unmelted.rename(index={'United States': 'US', "Czech Republic":"Czechia", "South Korea": "Korea, South"})



gr_confirmed1["Country/Region"].replace({'Taiwan*': 'Taiwan'}, inplace = True)

for df in df_unmelted:

    df["Country/Region"].replace({'United States': 'US', "Czech Republic":"Czechia", "South Korea": "Korea, South"}, inplace = True)

# Making the Changes in original datasets of Confirmed Cases/ Deaths/ Recovered  datasets and their melted forms



gr_confirmed = gr_confirmed.rename(index={'Taiwan*': 'Taiwan'})

gr_death = gr_death.rename(index={'Taiwan*': 'Taiwan'})

gr_recovered = gr_recovered.rename(index={'Taiwan*': 'Taiwan'})



gr_confirmed_melt = gr_confirmed_melt.replace({'Taiwan*': 'Taiwan'})

gr_death_melt = gr_death_melt.replace({'Taiwan*': 'Taiwan'})

gr_recovered_melt = gr_recovered_melt.replace({'Taiwan*': 'Taiwan'})
# To include all the countries which are there in Confirmed Cases/ Deaths/ Recovered Cases Datasets 

# By joining the "Country/Region"  column in Confirmed Cases dataset with the whole testing (unmelted) data by joining on

# common column "Country/Region"



# df_test_unmelted = pd.merge(gr_confirmed1['Country/Region'], df_test_unmelted, on='Country/Region', how='outer', indicator=True)

# df_test_unmelted = df_test_unmelted_1[df_test_unmelted_1['_merge'] == 'left_only']

# for df in df_unmelted:

for i in range(0,(len(df_unmelted))):

    df_unmelted[i] = pd.merge(gr_confirmed1['Country/Region'], df_unmelted[i], on='Country/Region', how='left')

# df_unmelted[2]["Country/Region"].unique()

df_unmelted[2].head()
# for df in df_unmelted:

#     df.fillna(0, inplace=True)

# #     df.set_index("Country/Region", inplace=True)

# df_unmelted[2].head(3)
for df in df_unmelted:

    df.set_index("Country/Region", inplace=True)
name = ["Cumulative Testing Count", "Daily Testing Count", "7-day smoothed daily change in testing", "7-day smoothed daily change per thousand in testing"]

df_melt=[]

for index, df in enumerate(df_unmelted):

    # Dates are converted into no of days since 1/1/20 so that 1/1/20 corresponds to day 1

    df_test_melt = df

    dates = df_test_melt.keys()

    FMT = '%m/%d/%y'



    days = dates.map(lambda x: (datetime.strptime(x, FMT) - datetime.strptime("01/01/20", FMT)).days)

    days = days + 1

    # +1 is to start the days from 1 instead of 0



    df_test_melt.columns = days # = dates will convert the columns to date formats again



    x_lag  # x_lag gives position of first value in each row which is greater than or equal to 5 in confirmed cases data



    for i in range(df_test_melt.shape[0]): # gr_recovered_melt.shape[0] = 187 (no. of rows) and gr_recovered_melt.shape[1] = 138 (no. of columns)

        df_test_melt.iloc[i] = df_test_melt.iloc[i].shift(periods=-x_lag[i]+1) # all data shift to one starting point 

    df_test_melt.head(3)



    #### Melting our Confirmed Cases dataset



    df_test_melt = df_test_melt.reset_index()

    df_test_melt = pd.melt(df_test_melt,id_vars= "Country/Region", 

                           value_vars=days, var_name="Days", 

                           value_name=name[index]).sort_values(["Country/Region","Days"], ignore_index=True)



#     df_test_melt.to_csv("C:/Users/user/Downloads/COVID-19 Related info/Created Files/Testing_daywise_melted.csv")

    df_melt.append(df_test_melt)

    
for i in range(1,(len(df_melt))): # loop is for len(data)-1 time, because upper bound is not included

    df_melt[0] = pd.merge(df_melt[0], df_melt[i], how="outer", on=["Country/Region", "Days"])



df_test_melt= df_melt[0]
# df_test_melt = df_test_melt.set_index("Country/Region")

df_test_melt.loc[df_test_melt["Country/Region"] =="Austria"].head(5)
df_test_melt["smoothed_cumulative_testing_count"] = df_test_melt["Cumulative Testing Count"].copy()

for country in df_test_melt["Country/Region"].unique():

    index_max = df_test_melt.loc[df_test_melt["Country/Region"]==country, "smoothed_cumulative_testing_count"].argmax()

    df_test_melt.loc[df_test_melt["Country/Region"]==country, "smoothed_cumulative_testing_count"] = df_test_melt.loc[df_test_melt["Country/Region"]==country, "smoothed_cumulative_testing_count"].iloc[:index_max+1].interpolate()
# Rounding of the values in the created new column "smoothed_cumulative_testing_count"

df_test_melt["smoothed_cumulative_testing_count"] = round(df_test_melt["smoothed_cumulative_testing_count"])



# Rearranging the columns such that "smoothed_cumulative_testing_count" is adjacent to "Cumulative conformed Count" column for validation.

df_test_melt = df_test_melt[["Country/Region", "Days", "Cumulative Testing Count", "smoothed_cumulative_testing_count",

                             "Daily Testing Count",

                            "7-day smoothed daily change in testing", "7-day smoothed daily change per thousand in testing"]]

# df_test_melt = df_test_melt.set_index("Country/Region")
# df_test_melt.loc[df_test_melt["Country/Region"] =="Germany"]
# gr_confirmed_melt.shape

# gr_death_melt.shape

# gr_recovered_melt.shape

# gr_recovered_melt.shape

# df_test_melt.shape
df1 = gr_confirmed_melt 

df2 = gr_death_melt 

df3 = gr_recovered_melt 

df4 = df_test_melt
# df1.to_csv("C:/Users/user/Downloads/COVID-19 Related info/Created Files/df1.csv")

# df2.to_csv("C:/Users/user/Downloads/COVID-19 Related info/Created Files/df2.csv")

# df3.to_csv("C:/Users/user/Downloads/COVID-19 Related info/Created Files/df3.csv")

# df4.to_csv("C:/Users/user/Downloads/COVID-19 Related info/Created Files/df4.csv")
data = [df1, df2, df3, df4]

len(data)

for i in range(1,(len(data))): # Actually the loop runs for len(data)-1 time, because upper bound is not included

    data[0] = pd.merge(data[0], data[i], how="inner", on=["Country/Region", "Days"])

MergedData = data[0]

# MergedData.set_index("Country/Region", inplace = True)
# MergedData.to_csv("C:/Users/user/Downloads/COVID-19 Related info/Created Files/Merged_Data.csv")
MergedData.head()
# df_pop = pd.read_csv("C:/Users/user/Downloads/COVID-19 Related info/Population Data.csv", encoding  = 'latin-1')

df_pop = pd.read_csv("../input/population-data/Population Data.csv", encoding = "latin-1", header = 0)

df_pop.head(3)

# df_pop.shape
# temp = pd.merge(MergedData, df_pop, how="outer", on=["Country/Region"], indicator = True)
# Check which countries are only in df_pop and which are only in MergedData and see if there are different names 

# for the same countries in the two dataframes and naming them identically



# temp[temp['_merge'] == 'right_only']

# temp[temp['_merge'] == 'left_only']
df_pop['Country/Region'] = df_pop['Country/Region'].replace({ 'Myanmar': 'Burma',

                                                               'Congo': 'Congo (Brazzaville)',

                                                               'DR Congo': 'Congo (Kinshasa)',

                                                               'Côte d\'Ivoire': 'Cote d\'Ivoire',

                                                               'Czech Republic (Czechia)': 'Czechia',

                                                              'South Korea': 'Korea, South',

                                                              'Saint Kitts & Nevis':'Saint Kitts and Nevis',

                                                              'St. Vincent & Grenadines':'Saint Vincent and the Grenadines',

                                                              'Sao Tome & Principe':'Sao Tome and Principe',

                                                              'United States': 'US',

                                                              'State of Palestine' : 'West Bank and Gaza',

                                                              })

MergedData = pd.merge(MergedData, df_pop, how="left", on=["Country/Region"])

MergedData.head(3)
# MergedData.loc[MergedData['Population'].isna()]
MergedData.loc[MergedData["Country/Region"] == 'Kosovo', 'Population'] = 1810936

MergedData.loc[MergedData["Country/Region"] == 'Kosovo', 'Land_Area_Kmsq'] = 10887

MergedData.loc[MergedData["Country/Region"] == 'Kosovo', 'Pop_Density'] = 166
# df_hcs = pd.read_csv("C:/Users/user/Downloads/COVID-19 Related info/Health Capacity Score.csv", encoding  = 'latin-1')

df_hcs = pd.read_csv("../input/health-capacity-score/Health Capacity Score.csv", header = 0, encoding = "latin-1")

df_hcs.head(3)
# df_hcs.sort_values(by = "Score/100", ascending = False).head(10)
temp1 = pd.merge(MergedData, df_hcs, how="outer", on=["Country/Region"], indicator = True)
# Check which countries are only in df_hi and which are only in MergedData and see if there are different names 

# for the same countries in the two dataframes and naming them identically



# temp1[temp1['_merge'] == 'right_only']

# temp1[temp1['_merge'] == 'left_only']
df_hcs['Country/Region'] = df_hcs['Country/Region'].replace({ 'Myanmar': 'Burma',

                                                               'Congo (Democratic Republic)': 'Congo (Kinshasa)',

                                                               'Côte d\'Ivoire': 'Cote d\'Ivoire',

                                                               'Czech Republic': 'Czechia',

                                                               'eSwatini (Swaziland)' : 'Eswatini',

                                                               'Kyrgyz Republic': 'Kyrgyzstan',

                                                              'South Korea': 'Korea, South',

                                                              'St Kitts and Nevis':'Saint Kitts and Nevis',

                                                              'St Lucia' : 'Saint Lucia',

                                                              'St Vincent and The Grenadines':'Saint Vincent and the Grenadines',

                                                              'São Tomé and Príncipe':'Sao Tome and Principe',

                                                              'United States': 'US',

                                                              'State of Palestine' : 'West Bank and Gaza'

                                                              })
MergedData = pd.merge(MergedData, df_hcs, how="left", on=["Country/Region"])

MergedData.head(3)
MergedData.loc[MergedData["Country/Region"] == "Australia"].head(3)


MergedData['Daily Confirmed Count'] = MergedData['Cumulative Confirmed Count'].diff()#.fillna(MergedData['Cumulative Confirmed Count'])

MergedData['Daily_Confirmed_7day_rolling_average'] = MergedData["Daily Confirmed Count"].rolling(window=7).mean()

MergedData['Daily_Confirmed_7day_rolling_average_per_million'] = (MergedData['Daily_Confirmed_7day_rolling_average']/MergedData["Population"])*1000000

MergedData['Daily Death Count'] = MergedData['Cumulative Death Count'].diff()#.fillna(MergedData["Cumulative Death Count"])

MergedData['Daily Recovered Count'] = MergedData['Cumulative Recovered Count']#.diff().fillna(MergedData["Cumulative Recovered Count"])



MergedData["RecoveryRate"] = round((MergedData["Cumulative Recovered Count"]/MergedData["Cumulative Confirmed Count"])*100,4)

MergedData["FatalityRate"] = round((MergedData["Cumulative Death Count"]/MergedData["Cumulative Confirmed Count"])*100,2)

MergedData["ActiveCases"] = MergedData["Cumulative Confirmed Count"]-MergedData["Cumulative Recovered Count"]-MergedData["Cumulative Death Count"]

# MergedData["CumConfirmed_per_Population"] = round((MergedData["Cumulative Confirmed Count"]/MergedData["Population"])*100,6)

# MergedData["CumTesting_per_Population"] = round(MergedData["Cumulative Testing Count"]/MergedData["Population"],4)

# MergedData["CumTesting_per_CumConfirmed"] = round(MergedData["Cumulative Testing Count"]/MergedData["Cumulative Confirmed Count"],4)

MergedData["CumConfirmed_per_million"] = round((MergedData["Cumulative Confirmed Count"]/MergedData["Population"])*1000000,6)

MergedData["SmoothedCumTesting_per_thousand"] = round((MergedData["smoothed_cumulative_testing_count"]/MergedData["Population"])*1000,6)

MergedData["SmoothedCumTesting_per_CumConfirmed"] = round(MergedData["smoothed_cumulative_testing_count"]/MergedData["Cumulative Confirmed Count"],4)

MergedData["CumConfirmed_per_SmoothedCumTesting_percent"] = round((MergedData["Cumulative Confirmed Count"]/MergedData["smoothed_cumulative_testing_count"])*100,4)







MergedData = MergedData[['Country/Region', 'Days', 'Cumulative Confirmed Count', 'Daily Confirmed Count', 

                         'Daily_Confirmed_7day_rolling_average', 'Daily_Confirmed_7day_rolling_average_per_million',

                         'Cumulative Death Count', 'Daily Death Count', 

                         'Cumulative Recovered Count', 'Daily Recovered Count', 

                         'Cumulative Testing Count','smoothed_cumulative_testing_count', 'Daily Testing Count',

                         '7-day smoothed daily change in testing', '7-day smoothed daily change per thousand in testing', 

                         'Population', 'Pop_Density','Land_Area_Kmsq', 

                         'Score/100',

                         'RecoveryRate','FatalityRate', 'ActiveCases',

                         'CumConfirmed_per_million', 'SmoothedCumTesting_per_thousand',

                         'SmoothedCumTesting_per_CumConfirmed', "CumConfirmed_per_SmoothedCumTesting_percent"

                        ]] 

MergedData.head(5)
#### Interpolating "Cumulative Testing Count " and storing the values in a new column named "smoothed_cumulative_testing_count"
# MergedData["smoothed_cumulative_testing_count"] = MergedData["Cumulative Testing Count"].copy()

# for country in MergedData["Country/Region"].unique():

#     index_max = MergedData.loc[df_test_melt["Country/Region"]==country, "Cumulative Confirmed Count"].argmax()

#     MergedData.loc[MergedData["Country/Region"]==country, "smoothed_cumulative_testing_count"] = MergedData.loc[MergedData["Country/Region"]==country, "smoothed_cumulative_testing_count"].iloc[:index_max+1].interpolate()



# # Rounding of the values in the created new column "smoothed_cumulative_testing_count"

# MergedData["smoothed_cumulative_testing_count"] = round(MergedData["smoothed_cumulative_testing_count"])
MergedData.columns
# MergedData.loc[MergedData["Country/Region"]=="Fiji"][["Country/Region", "Days","Daily Confirmed Count", "Cumulative Confirmed Count",

#                                                        "Cumulative Testing Count","smoothed_cumulative_testing_count", "SmoothedCumTesting_per_CumConfirmed",

#                                                          "Cumulative Testing Count", "CumTesting_per_CumConfirmed"]]
#output_file-to save the layout in file, show-display the layout , output_notebook-to configure the default output state  to generate the output in jupytor notebook.

from bokeh.io import curdoc, output_file, show , output_notebook 

#ColumnDataSource makes selection of the column easier and Select is used to create drop down 

from bokeh.models import ColumnDataSource, DataRange1d, Legend, Div, Select, RadioGroup, MultiSelect, Title, Label, Span

from bokeh.models import ZoomInTool,ZoomOutTool, ResetTool, HoverTool, BoxZoomTool

from bokeh.models import CustomJS, Panel, Tabs, BoxAnnotation

from bokeh.plotting import figure 

from bokeh.layouts import column, row, gridplot, layout

from bokeh.palettes import YlOrRd, RdYlGn, Inferno, Viridis, Magma

output_notebook() #create default state to generate the output

# Creating a copy of MergedData to be used for converting it to ColumnDataSources for bokeh 

# plots

data = MergedData.copy()
# Spaced column names do not returend values while hovering over bokeh plot, so creating one word 

# names for columns



data.rename(columns={'Country/Region': 'Country',

                     'Cumulative Confirmed Count':'CumConfirmed',

                     'Cumulative Death Count':'CumDeaths',

                     'Cumulative Recovered Count':'CumRecovered',

                     'Cumulative Testing Count':'CumTesting',

                     'Daily Confirmed Count':'DailyConfirmed',

                     'Daily Death Count':'DailyDeaths',

                     'Daily Recovered Count':'DailyRecovered',

                     'Daily Testing Count':'DailyTesting', 

                     '7-day smoothed daily change in testing': 'Daily_testing_7day_rolling_average',

                     '7-day smoothed daily change per thousand in testing': 'Daily_testing_7day_rolling_average_per_thousand',

                     'ActiveCases': 'Active',

                     'Score/100': 'HealthCapacityScore'}, 

                      inplace=True)
# Step 1: Creating ColumnDataSources 



data_all=data.loc[:, ['Country','Days', 'CumConfirmed', 'CumDeaths', 'CumRecovered', 'CumTesting', 'smoothed_cumulative_testing_count']]

data_curr = data_all[data_all['Country'] == 'Australia' ]



Overall = ColumnDataSource(data=data_all)

Curr=ColumnDataSource(data=data_curr)



#------------------------------------------------------------------------------------------



# Step 2:

# Defining callback function which links plots and the select menu



callback = CustomJS(args=dict(source=Overall, current=Curr), code="""

var selected_country = cb_obj.value

current.data['Days']=[]

current.data['CumConfirmed']=[]

current.data['CumDeaths'] = []

current.data['CumRecovered'] = []

current.data['CumTesting'] = []

current.data["smoothed_cumulative_testing_count"] = []

for(var i = 0; i <= source.get_length(); i++){

	if (source.data['Country'][i] == selected_country){

		current.data['Days'].push(source.data['Days'][i])

		current.data['CumConfirmed'].push(source.data['CumConfirmed'][i])

		current.data['CumDeaths'].push(source.data['CumDeaths'][i])

		current.data['CumRecovered'].push(source.data['CumRecovered'][i])

		current.data['CumTesting'].push(source.data['CumTesting'][i]) 

        current.data["smoothed_cumulative_testing_count"].push(source.data["smoothed_cumulative_testing_count"][i])

	 }          

} 

current.change.emit();

""")



#-------------------------------------------------------------------------------------------

# Step 3: Creating menu

menu = Select(options=list(data['Country'].unique()),value='Australia', title="Select Country:")  # drop down menu



#------------------------------------------------------------------------------------------

# Step 4: Creating plot



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot

P = [] 

axis_type =["log", "linear"]



# 2.

# Creating figure for the plot

for i in range(2):

    p1 = figure(x_axis_label ='Days', y_axis_label = 'Number of Cases', y_axis_type = axis_type[i])#creating figure object 



# 3.

# Plotting line graph on the figure

       

    line1 = p1.line(x='Days', y="CumConfirmed", source=Curr, color = "red", line_width = 2) # plotting the data using glyph circle

    circle1 = p1.circle(x='Days', y="CumConfirmed", source=Curr, color = "red", size = 1)

    line2 = p1.line(x='Days', y="CumDeaths", source=Curr, color = "black", line_width = 2)

    circle2 = p1.circle(x='Days', y="CumDeaths", source=Curr, color = "black", size = 2)

    line3 = p1.line(x='Days', y="CumRecovered", source=Curr, color = "darkgreen", line_width = 2)

    circle3 = p1.circle(x='Days', y="CumRecovered", source=Curr, color = "darkgreen", size = 2)

    line4 = p1.line(x='Days', y="smoothed_cumulative_testing_count", source=Curr, color = "navy", line_width = 2)

    circle4 = p1.circle(x='Days', y="smoothed_cumulative_testing_count", source=Curr, color = "navy", size = 2)  



# 4.

# Stylizing



    # Stylize the plot area

    p1.plot_width = 650               

    p1.plot_height = 350                 

    p1.background_fill_color = "#1f77b4"   

    p1.background_fill_alpha = 0.12

    

    # Stylize the grid

    p1.xgrid.grid_line_color = "white"

    p1.ygrid.grid_line_color = "white"

    p1.ygrid.grid_line_alpha = 0.7

    p1.grid.grid_line_dash = [5,3]

    

    # Axes Geometry

    p1.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p1.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

    

    # Stylize the axes

#     p1.y_range.renderers = [line1]

    if axis_type[i] == "log":

        p1.y_range = DataRange1d(start=1, end = 300000000)

    if axis_type[i] == "linear":

        p1.yaxis.formatter.use_scientific = False

    p1.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p1.axis.axis_label_text_color = "black"

    p1.yaxis.minor_tick_in = 0

    p1.yaxis.minor_tick_out = 0

    p1.xaxis.minor_tick_in = 0

    p1.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p1.title.text_color = "black"

    p1.title.text_font = "times"

    p1.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p1.title.align = "center"



    # Stylize the tools

    # Adding customization to HoverTool

#     p1.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [(("Day", "@Days")),("Cumulative Confirmed Count", "@CumConfirmed"), ("Cumulative Death Count", "@CumDeaths"), ("Cumulative Recovered Count", "@CumRecovered"), ("Cumulative Testing Count", "@smoothed_cumulative_testing_count")])

    p1.add_tools(hover) # Customized HoverTool

    

    # Stylize the legends

    legend = Legend(items=[

    ("Cum Confirmed Count",   [line1, circle1]),

    ("Cum Death Count", [line2, circle2]),

    ("Cum Recovered Count", [line3, circle3]),

    ("Cum Testing Count ",[line4, circle4]),    

     ]) #, location=(10,80))

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0.01

    legend.label_text_font = "times"

    legend.label_text_font_size = '12px'

    legend.title = "Select/Unselect:"

    legend.click_policy="hide"  # To disable/hide the legend on click

    # Adding the legends as a layout to the figure

    p1.add_layout(legend, 'right')

    

    # Appending the two figures created under for loop (one with y_axis_type="log" and other

    # with y_axis_type = "linear") into the list P

    P.append(p1)



# 5. 

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Log scale")

tab2 = Panel(child=P[1], title="Linear scale")



#------------------------------------------------------------------------------------------

# Step 5: Creating list of Tabs

tabs = Tabs(tabs=[ tab1, tab2 ])



#------------------------------------------------------------------------------------------

# Step 6: Calling the function on change of selection



menu.js_on_change('value', callback) 



#------------------------------------------------------------------------------------------

# Step 7: Adding title and footer as a <div> tag (a division or a section in an HTML document)



title = Div(text = "<b> Cumulative Confirmed , Deaths, Recovered and Testing counts for individual countries </b>", 

            width=600, style={'font-size': '125%', 'color': 'black'}, align = "center")

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 



#------------------------------------------------------------------------------------------

# Step 8: Creating the layout



layout=column(title,menu,tabs, footer) 



#------------------------------------------------------------------------------------------

# Step 9: Displaying the layout

show(layout) 

# Creating plots



# A)

#------------------------------------------------------------------------------------------

# Daily Confirmed Count plot for top 5 Health Capacity score countries

#------------------------------------------------------------------------------------------



# 1. 

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P = [] 

axis_type =["log", "linear"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Cumulative Confirmed Count", y_axis_type =axis_type[i])

    # creating figure object 

    #, plot_width=850, plot_height=400



# 3.

# Creating the graph on figure



    circle1 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", line_width = 2.5) # plotting the data using glyph circle

    

    circle2 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", line_width = 2.5) # plotting the data using glyph circle



    circle3 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", line_width = 2.5) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", line_width = 2.5) # plotting the data using glyph circle

    

    circle5 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", size = 2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", line_width = 2.5) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 2.5) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", line_width = 2.5) # plotting the data using glyph circle



    circle8 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", size = 2) # plotting the data using glyph circle

    line8 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", line_width = 2.5) # plotting the data using glyph circle



# 4. 

# Stylizing

    

    # Stylize the plot area

    p3.plot_width = 600                

    p3.plot_height = 350                

    p3.background_fill_color = "#1f77b4"   

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(start=10, end = 1000000)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

    

    

    # Stylize the axes

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

#     p3.y_range.start = 0

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customizayion to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}")

                                   ])



    p3.add_tools(hover) # Customization of HoverTool

    

    # Adding Annotations

    mytext1 = Label(x=187, y=25000, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)



    # Stylize Legends

    legend = (Legend(items=[("Germany", [circle1, line1]),

                            ("Korea, South", [circle2, line2]),

                            ("Finland", [circle3, line3]),

                            ("Denmark", [circle4, line4]),

                            ("Netherlands", [circle5, line5]),

                            ("Australia", [circle6, line6]),

                            ("Croatia", [circle7, line7]),

                            ("Japan", [circle8, line8])

                            ]))

    

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")

    P.append(p3)

    

# 5. 

# Creating separate tabs for the two figures created above

tab11 = Panel(child=P[0], title="Log scale")

tab12 = Panel(child=P[1], title="Linear scale")



# 6.

# Creating list of Tabs

tabs1 = Tabs(tabs=[tab11, tab12])



# 7.

# Adding title as a <div> tag (a division or a section in an HTML document)



title1 = Div(text = "<b> Trajectories for <i> Cumulative Confirmed Counts</i> for Top 8 countries with highest Health Capacity Scores </b>", 

           style={'font-size': '125%', 'color': 'black'}, align = "start")

footer1 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")





# B) 

#------------------------------------------------------------------------------------------

# Daily Testing Count plot for top 5 Health Capcity score countries

#------------------------------------------------------------------------------------------



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P = [] 

axis_type =["log", "linear"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Cumulative Testing Count", y_axis_type =axis_type[i])

#                 , plot_width=850, plot_height=400)#creating figure object 





# 3.

# Creating the graph on figure 



    circle1 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", line_width = 2.5) # plotting the data using glyph circle

    

    circle2 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", line_width = 2.5) # plotting the data using glyph circle



    circle3 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", line_width = 2.5) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", line_width = 2.5) # plotting the data using glyph circle



    circle5 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", size = 2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", line_width = 2.5) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 2.5) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", line_width = 2.5) # plotting the data using glyph circle



    circle8 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", size = 2) # plotting the data using glyph circle

    line8 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", line_width = 2.5) # plotting the data using glyph circle



    

# 4. 

# Stylizing

   

    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 400                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(start=100, end = 50000000)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

    

    

    # Stylize the axes

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

#     p3.y_range.start = 0

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customization to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}")

                                   ])



    p3.add_tools(hover) # Customized HoverTool

    

    # Adding Annotations

    mytext1 = Label(x=187, y=5500000, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)



    # Stylize Legends

    legend = (Legend(items=[("Germany", [circle1, line1]),

                            ("Korea, South", [circle2, line2]),

                            ("Finland", [circle3, line3]),

                            ("Denmark", [circle4, line4]),

                            ("Netherlands", [circle5, line5]),

                            ("Australia", [circle6, line6]),

                            ("Croatia", [circle7, line7]),

                            ("Japan", [circle8, line8])

                        ]))

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")



# 5. 

# Appending the figures for log and linear scale for y-axis

    P.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab21 = Panel(child=P[0], title="Log scale")

tab22 = Panel(child=P[1], title="Linear scale")



# 7.

# Creating list of Tabs

tabs2 = Tabs(tabs=[tab21, tab22])



# 8.

# Adding title as a <div> tag (a division or a section in an HTML document)



title2 = Div(text = "<b> Trajectories for <i> Cumulative Testing Counts</i> for Top 8 countries with highest Health Capacity Scores </b>", 

            style={'font-size': '125%', 'color': 'black'}, align = "start")



#------------------------------------------------------------------------------------------

# Adding common footers as <div> tags (a division or a section in an HTML document)

footer2 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

footer3 = Div(text = "Health Capacity Score is a highly relevant attribute of the comprehensive Global Health Security Index relating to country's <br> preparedness for COVID-19 <i>(https://www.ghsindex.org)</i>", 

            style={'font-size': '100%', 'color': 'black'}, align = "center") 



#------------------------------------------------------------------------------------------

# Creating and showing the layout 

layout = column(title1, tabs1, footer1, title2, tabs2, footer2, footer3)

show(layout)
# Creating Plots



# A)

#------------------------------------------------------------------------------------------

# Daily Confirmed Count plot for top 7 highly populated countries and Australia

#------------------------------------------------------------------------------------------



# 1. 

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot

P = [] 

axis_type =["log", "linear"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Cumulative Confirmed Count", y_axis_type =axis_type[i])

#                 , plot_width=850, plot_height=400)#creating figure object 



# 3. 

#  Creating the graphs on figure



    circle1 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", line_width =2.5 ) # plotting the data using glyph circle



    circle2 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", line_width = 2.5) # plotting the data using glyph circle

    

    circle3 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", line_width = 2.5) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", line_width = 2.5) # plotting the data using glyph circle



    circle5 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", size = 2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", line_width = 2.5) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", size = 2) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", line_width = 2.5) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", line_width = 2.5) # plotting the data using glyph circle



    circle = p3.circle(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 2.5) # plotting the data using glyph circle

    line = p3.line(x='Days', y="CumConfirmed", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

#     p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(start=10, end = 10000000)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

        

    # Stylize the axes

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customizayion to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("Cumulative Testing Count", "@CumTesting{int}"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}"),

                                  ("Confirmed this day", "@DailyConfirmed"),

                                  ("Tested this day", "@DailyTesting{int}"), 

                                   ])



    p3.add_tools(hover) # Customization of HoverTool



     # Adding Annotations

    mytext1 = Label(x=187, y=25000, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)

   



    # Stylize Legends

    legend = (Legend(items=[("China", [circle1, line1]),

                            ("India", [circle2, line2]),

                            ("US", [circle3, line3]),

                            ("Indonesia", [circle4, line4]),

                            ("Pakistan", [circle5, line5]),

                            ("Brazil", [circle6, line6]),

                            ("Nigeria", [circle7, line7]),

                            ("Australia", [circle, line])

                            ]))

    

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")



# 5. 

# Appending the figures with log and linear scale for y-axis

    P.append(p3)

    

# 6. 

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Log scale")

tab2 = Panel(child=P[1], title="Linear scale")



# 7.

# Creating list of Tabs

tabs_confirmed = Tabs(tabs=[ tab1, tab2 ])



# 8.

# Adding title and footer as a <div> tag (a division or a section in an HTML document)

title1 = Div(text = "<b> Comparison for <i>Cumulative Confirmed Counts'</i> trajectories for<br>Top 7 highly populated countries and Australia</b>", 

            style={'font-size': '125%', 'color': 'black'}, align = "start")#width=600, 



footer1 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

# 9.

# Creating the layout

layout1=column(title1, tabs_confirmed, footer1)



# B)

#------------------------------------------------------------------------------------------

# Cumulative testing Count plot for top 7 highly populated countries and Australia

#------------------------------------------------------------------------------------------



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P = [] 

axis_type =["log", "linear"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Testing Count", y_axis_type =axis_type[i])

#     , plot_width=850, plot_height=400)#creating figure object 





# 3.

# Creating the graph on figure





    circle1 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", line_width = 2.5) # plotting the data using glyph circle



    circle2 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", line_width = 2.5) # plotting the data using glyph circle

    

    circle3 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", line_width = 2.5) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", line_width = 2.5) # plotting the data using glyph circle



    circle5 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", size = 2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", line_width = 2.5) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", size = 2) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", line_width = 2.5) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", line_width = 2.5) # plotting the data using glyph circle



    circle = p3.circle(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 2.5) # plotting the data using glyph circle

    line = p3.line(x='Days', y="smoothed_cumulative_testing_count", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

#     p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(start=100, end = 300000000)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

      

    # Stylize the axes

    # p3.yaxis.formatter.use_scientific = False

    # p3.xaxis.formatter.use_scientific = False

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customizayion to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("Cumulative Testing Count", "@CumTesting{int}"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}"),

                                  ("Confirmed this day", "@DailyTesting"),

                                  ("Tested this day", "@DailyTesting{int}")

                                   ])



    p3.add_tools(hover) # Customization of HoverTool

    

    # Adding Annotations

    mytext1 = Label(x=187, y=5500000, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)





    #Stylize Legends

    legend = (Legend(items=[("China", [circle1, line1]),

                            ("India", [circle2, line2]),

                            ("US", [circle3, line3]),

                            ("Indonesia", [circle4, line4]),

                            ("Pakistan", [circle5, line5]),

                            ("Brazil", [circle6, line6]),

                            ("Nigeria", [circle7, line7]),

                            ("Australia", [circle, line])

                            ]))

        

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")



# 5. 

# Appending the figures for log and linear scale for y-axis

    P.append(p3)

    

# 6. 

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Log scale")

tab2 = Panel(child=P[1], title="Linear scale")



# 7.

# Creating list of Tabs

tabs_testing = Tabs(tabs=[ tab1, tab2 ])



# 8.

# Adding title and footer as a <div> tag (a division or a section in an HTML document)

title2 = Div(text = "<b> Comparison of <i>Cumulative Testing Counts'</i> trajectories for <br>Top 7 highly Populated countries and Australia</b>", 

           style={'font-size': '125%', 'color': 'black'}, align = "start") # width=600, 



footer2 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

# 9.

# Creating the layout

layout2=column(title2, tabs_testing, footer2)



#------------------------------------------------------------------------------------------

# Creating the final layout and displaying

layout = column(layout1, layout2)



# Displaying the layout

show(layout) 

# Step 1: Creating ColumnDataSources 



data_all=data.loc[:, ['Country','Days', 'CumConfirmed', 'CumRecovered', 'CumDeaths', 'Active', 'CumTesting', 'RecoveryRate', "Population", "Pop_Density"]]



Overall = ColumnDataSource(data=data_all)

Curr = ColumnDataSource(dict(Country = [], Days = [], CumConfirmed = [], CumRecovered = [], CumDeaths = [], Active = [], CumTesting = [], RecoveryRate = [], Population = [], Pop_Density = []))



#------------------------------------------------------------------------------------------



# Step 2:

# Defining callback function which links plots and the select menu



multi_countries_callback = CustomJS(args=dict(source=Overall, current=Curr), code="""

var Selected_Country = cb_obj.value



current.data['Country'] = []

current.data['Days']=[]

current.data['CumConfirmed'] = []

current.data['CumRecovered'] = []

current.data['CumDeaths'] = []

current.data['Active'] = []

current.data['CumTesting'] = []

current.data['RecoveryRate'] = []

current.data['Population'] = []

current.data['Pop_Density'] = []





for(var i = 0; i <= source.get_length(); i++){

	if (Selected_Country.indexOf(source.data['Country'][i]) >= 0){

        current.data['Country'].push(source.data['Country'][i])

		current.data['Days'].push(source.data['Days'][i])

        current.data['CumConfirmed'].push(source.data['CumConfirmed'][i])

        current.data['CumRecovered'].push(source.data['CumRecovered'][i])

        current.data['CumDeaths'].push(source.data['CumDeaths'][i])

        current.data['Active'].push(source.data['Active'][i])

        current.data['CumTesting'].push(source.data['CumTesting'][i])

		current.data['RecoveryRate'].push(source.data['RecoveryRate'][i])

        current.data['Population'].push(source.data['Population'][i])

        current.data['Pop_Density'].push(source.data['Pop_Density'][i])

        

	 }          

} 

current.change.emit();

""")



#-------------------------------------------------------------------------------------------

# Step 3: Creating menu



menu = MultiSelect(options=list(data['Country'].unique()),value=[], title= "Select one or more Country/Region:") # drop down menu



#------------------------------------------------------------------------------------------

# Step 4: Creating plot



# 1.

# Creating figure for the plot



p2 = figure(x_axis_label ='Days', y_axis_label = 'Recovery Rate', y_axis_type ="linear")#creating figure object 



# 2.

# Creating the graph on figure

line = p2.line(x='Days', y= "RecoveryRate", source=Curr, color = "black", line_width = 2) # plotting the data using glyph circle





# 3. Stylizing



# Stylize the plot area

p2.plot_width = 600                # To change the width of the plot

p2.plot_height = 350                 # To change the height of plot

p2.ygrid.grid_line_alpha = 0.7



# Axes Geometry

# p2.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

# p2.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

p2.y_range.start = 0

p2.x_range.start = 0



# Stylize the axes

p2.axis.axis_label_text_font_style = "bold" 

p2.axis.axis_label_text_color = "black"

p2.yaxis.minor_tick_in = 0

p2.yaxis.minor_tick_out = 0

p2.xaxis.minor_tick_in = 0

p2.xaxis.minor_tick_out = 0





# Box Annotations

verylow_box = BoxAnnotation(bottom = 0, top= 25, fill_color=RdYlGn[11][7])

low_box = BoxAnnotation(bottom=25, top=50, fill_color=RdYlGn[11][4])

medium_box = BoxAnnotation(bottom=50, top=75, fill_color=RdYlGn[11][4])

high_box = BoxAnnotation(bottom=75,top = 100, fill_color=RdYlGn[11][3]) # fill_alpha=0.18,



p2.add_layout(verylow_box)

p2.add_layout(low_box)

p2.add_layout(medium_box)

p2.add_layout(high_box)



# Adding spans to the figure

span_25 = Span(location=25, dimension='width', line_color='tomato', line_dash=[3,3])

span_75 = Span(location=75, dimension='width', line_color='green', line_dash=[3,3])

# Adding annotations to the spans

mytext_span25 = Label(x=0, y=25, text='25%', text_color = "black", text_font_size='8pt')

mytext_span75 = Label(x=0, y=75, text='75%', text_color = "black", text_font_size='8pt')

p2.add_layout(span_25)

p2.add_layout(span_75)

p2.add_layout(mytext_span25)

p2.add_layout(mytext_span75)



# Stylize the figure title

p2.title.text_color = "black"

p2.title.text_font = "times"

p2.title.text_font_size = "20px" # px stands for pixel. Have to mention.

p2.title.align = "center"



# Stylize the tools

# Adding customization to HoverTool

p2.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),("Total Confirmed", "@CumConfirmed"),("Total Recovered", "@CumRecovered"),("Recovery Rate", "@RecoveryRate{0.00}")])



p2.add_tools(hover) # Customization of HoverTool



#------------------------------------------------------------------------------------------

# Step 5: Calling the function on change of selection

menu.js_on_change('value', multi_countries_callback) 



#------------------------------------------------------------------------------------------

# Step 6: Adding title, subtitle and footer as <div> tags (a division or a section in an HTML document)



title = Div(text = "<b> Recovery Rate trajectory for individual countries </b>", 

            width=600, style={'font-size': '125%', 'color': 'black'}, align = "center")

subtitle = Div(text = "<i> Recovery Rate = (Cumulative Recovered / Cumulative Confirmed)*100 </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 



#------------------------------------------------------------------------------------------

# Step 7: Creating the layout

layout=column(title, subtitle, menu, p2, footer)



#------------------------------------------------------------------------------------------

# Step 8: Displaying the layout

show(layout) 



# 1.

# Creating figure for the plot



p2 = figure(x_axis_label ='Days', y_axis_label = 'Recovery Rate (%)', y_axis_type ="linear")

# , plot_width=850, plot_height = 400)#creating figure object 



# 2.

# Creating the graph on figure

line1 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "limegreen", line_width = 2) # plotting the data using glyph circle

line2 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle

line3 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", line_width = 2) # plotting the data using glyph circle

line4 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", line_width = 2) # plotting the data using glyph circle

line5 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", line_width = 2) # plotting the data using glyph circle

line6 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle

circle6 = p2.circle(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 3) # plotting the data using glyph circle

line7 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle

line8 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", line_width = 2) # plotting the data using glyph circle

line9 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Ireland"]), color = "mediumvioletred", line_width = 2) # plotting the data using glyph circle

line10 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Belarus"]), color = "gold", line_width = 2) # plotting the data using glyph circle



# 3.

# Stylizing



# Stylize the plot area

p2.ygrid.grid_line_alpha = 0.7

p2.plot_width=600

p2.plot_height = 350



# Axes Geometry

p2.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

p2.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

# p2.y_range.start = 0

# p2.x_range.start = 0



# Stylize the axes

p2.axis.axis_label_text_font_style = "bold" 

p2.axis.axis_label_text_color = "black"

p2.yaxis.formatter.use_scientific = False

p2.yaxis.minor_tick_in = 0

p2.yaxis.minor_tick_out = 0

p2.xaxis.minor_tick_in = 0

p2.xaxis.minor_tick_out = 0





# Box Annotations

verylow_box = BoxAnnotation(bottom = 0, top= 25, fill_color=RdYlGn[11][7])

low_box = BoxAnnotation(bottom=25, top=50, fill_color=RdYlGn[11][4])

medium_box = BoxAnnotation(bottom=50, top=75, fill_color=RdYlGn[11][4])

high_box = BoxAnnotation(bottom=75,top = 100, fill_color=RdYlGn[11][3]) # fill_alpha=0.18,



p2.add_layout(verylow_box)

p2.add_layout(low_box)

p2.add_layout(medium_box)

p2.add_layout(high_box)



# Adding spans to the figure

span_25 = Span(location=25, dimension='width', line_color='tomato', line_dash=[3,3])

span_75 = Span(location=75, dimension='width', line_color='green', line_dash=[3,3])

# Adding annotations to the spans

mytext_span25 = Label(x=0, y=25, text='25%', text_color = "black", text_font_size='8pt')

mytext_span75 = Label(x=0, y=75, text='75%', text_color = "black", text_font_size='8pt')

p2.add_layout(span_25)

p2.add_layout(span_75)

p2.add_layout(mytext_span25)

p2.add_layout(mytext_span75)



# Stylize the figure title

p2.title.text_color = "black"

p2.title.text_font = "times"

p2.title.text_font_size = "20px" # px stands for pixel

p2.title.align = "center"



# Stylize the tools

# Adding customization to HoverTool

p2.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),("Total Confirmed", "@CumConfirmed"),("Total Recovered", "@CumRecovered"),("Recovery Rate", "@RecoveryRate{0.00}%")])



p2.add_tools(hover) # Customized HoverTool



# Stylize Legends

legend = (Legend(items=[("Germany", [line1]),

                        ("Korea, South", [line2]),

                        ("Finland", [line3]),

                        ("Denmark", [line4]),

                        ("Netherlands", [line5]),

                        ("Australia", [line6, circle6]),

                        ("Croatia", [line7]),

                        ("Japan", [line8]),

                        ("Ireland", [line9]),

                        ("Belarus", [line10])

                       ]))

legend.background_fill_color = "#1f77b4"

legend.background_fill_alpha = 0

legend.border_line_color = None

legend.click_policy="hide"  # To disable/hide the legend on click

p2.add_layout(legend, "right")



# 4. 

# Adding Annotations to the plot

mytext1 = Label(x=192, y=50, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

p2.add_layout(mytext1)



# 5. 

# Adding title subtitle and footer as a <div> tag (a division or a section in an HTML document)



title = Div(text = "<b> Recovery Rate trajectories for Top 10 countries with the highest Health Capacity Scores </b>", 

            width=700, style={'font-size': '125%', 'color': 'black'}, align = "start")

# subtitle1 = Div(text = "Health Capacity Score is a highly relevant attribute of the comprehensive Global Health Security Index relating to country's preparedness for COVID-19 <i>(https://www.ghsindex.org)</i>", 

#             width=620, style={'font-size': '100%', 'color': 'black'}, align = "start") 

subtitle2 = Div(text = "<i> Recovery Rate = (Cumulative Recovered / Cumulative Confirmed)*100 </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "start") 

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 





# 6. 

# Creating the layout

layout=column(title, subtitle2, p2, footer) #subtitle1,



# 7. 

# Displaying the layout

show(layout) 

# 1.

# Creating figure for the plot



p2 = figure(x_axis_label ='Days', y_axis_label = 'Recovery Rate (%)', y_axis_type ="linear")

#             , plot_width=850, plot_height = 400)#creating figure object 



# 2.

# Creating the graph on figure

line1 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", line_width = 2) # plotting the data using glyph circle

line2 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", line_width = 2) # plotting the data using glyph circle

line3 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", line_width = 2) # plotting the data using glyph circle

line4 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", line_width = 2) # plotting the data using glyph circle

line5 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", line_width = 2) # plotting the data using glyph circle

line6 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle

line7 = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle

line = p2.line(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle

circle = p2.circle(x='Days', y="RecoveryRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 3) # plotting the data using glyph circle



# 3.

# Stylizing



# Stylize the plot area

p2.ygrid.grid_line_alpha = 0.7

p2.plot_width = 600

p2.plot_height = 350



# Axes Geometry

p2.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

p2.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

# p2.y_range.start = 0

# p2.x_range.start = 0



# Stylize the axes

p2.axis.axis_label_text_font_style = "bold" 

p2.axis.axis_label_text_color = "black"

p2.yaxis.minor_tick_in = 0

p2.yaxis.minor_tick_out = 0

p2.xaxis.minor_tick_in = 0

p2.xaxis.minor_tick_out = 0



# Box Annotations

verylow_box = BoxAnnotation(bottom = 0, top= 25, fill_color=RdYlGn[11][7])

low_box = BoxAnnotation(bottom=25, top=50, fill_color=RdYlGn[11][4])

medium_box = BoxAnnotation(bottom=50, top=75, fill_color=RdYlGn[11][4])

high_box = BoxAnnotation(bottom=75,top = 100, fill_color=RdYlGn[11][3]) # fill_alpha=0.18,



p2.add_layout(verylow_box)

p2.add_layout(low_box)

p2.add_layout(medium_box)

p2.add_layout(high_box)



# Adding spans to the figure

span_25 = Span(location=25, dimension='width', line_color='tomato', line_dash=[3,3])

span_75 = Span(location=75, dimension='width', line_color='green', line_dash=[3,3])

# Adding Annotations to spans

mytext_span25 = Label(x=0, y=25, text='25%', text_color = "black", text_font_size='8pt')

mytext_span75 = Label(x=0, y=75, text='100%', text_color = "black", text_font_size='8pt')

p2.add_layout(span_25)

p2.add_layout(span_75)

p2.add_layout(mytext_span25)

p2.add_layout(mytext_span75)



# Adding Annotations to the plot

mytext1 = Label(x=195, y=82, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

p2.add_layout(mytext1)



# Stylize the figure title

p2.title.text_color = "black"

p2.title.text_font = "times"

p2.title.text_font_size = "20px" # px stands for pixel. Have to mention.

p2.title.align = "center"



# Stylize the tools

# Adding customization to HoverTool

p2.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),("Total Confirmed", "@CumConfirmed"),("Total Recovered", "@CumRecovered"),("Recovery Rate", "@RecoveryRate{0.00}")])



p2.add_tools(hover) # Customization of HoverTool



# Stylize Legends

legend = (Legend(items=[("China", [line1]),

                        ("India", [line2]),

                        ("US", [line3]),

                        ("Indonesia", [line4]),

                        ("Pakistan", [line5]),

                        ("Brazil", [line6]),

                        ("Nigeria", [line7]),

                        ("Australia", [line, circle])

                        ]))

legend.background_fill_color = "#1f77b4"

legend.background_fill_alpha = 0

legend.border_line_color = None

legend.click_policy="hide"  # To disable/hide the legend on click

p2.add_layout(legend, "right")



# 4.

# Adding title subtitle and footer as a <div> tag (a division or a section in an HTML document)



title = Div(text = "<b> Comparison of Recovery Rate trajectories for <br>Top 7 highly populated countries and moderately populated Australia </b>", 

            width=600, style={'font-size': '125%', 'color': 'black'}, align = "start")

subtitle = Div(text = "<i> Recovery Rate = Cumulative Recovered / Cumulative Confirmed </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "start") 

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 



# 5.

# Creating the layout

layout1=column(title, subtitle, p2, footer)



# 6.

# Displaying the layout

show(layout1) 



# 1.

# Creating figure for the plot



p2 = figure(x_axis_label ='Days', y_axis_label = 'Fatality Rate (%)', y_axis_type ="linear")

# , plot_width=850, plot_height = 400)#creating figure object 



# 2.

# Creating the graph on figure

line1 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "limegreen", line_width = 2) # plotting the data using glyph circle

line2 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle

line3 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", line_width = 2) # plotting the data using glyph circle

line4 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", line_width = 2) # plotting the data using glyph circle

line5 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", line_width = 2) # plotting the data using glyph circle

line6 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle

circle6 = p2.circle(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 3) # plotting the data using glyph circle

line7 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle

line8 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", line_width = 2) # plotting the data using glyph circle

line9 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Ireland"]), color = "mediumvioletred", line_width = 2) # plotting the data using glyph circle

line10 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Belarus"]), color = "gold", line_width = 2) # plotting the data using glyph circle



# 3.

# Stylizing



# Stylize the plot area

p2.ygrid.grid_line_alpha = 0.7

p2.plot_width=600

p2.plot_height = 350



# Axes Geometry

p2.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

p2.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

# p2.y_range.start = 0

# p2.x_range.start = 0



# Stylize the axes

p2.axis.axis_label_text_font_style = "bold" 

p2.axis.axis_label_text_color = "black"

p2.yaxis.formatter.use_scientific = False

p2.yaxis.minor_tick_in = 0

p2.yaxis.minor_tick_out = 0

p2.xaxis.minor_tick_in = 0

p2.xaxis.minor_tick_out = 0





# Box Annotations

# verylow_box = BoxAnnotation(bottom = 0, top= 1, fill_color=RdYlGn[11][3])

low_box = BoxAnnotation(bottom=0, top=10, fill_color=RdYlGn[11][4])

# medium_box = BoxAnnotation(bottom=5, top=10, fill_color=RdYlGn[11][4])

high_box = BoxAnnotation(bottom=10, fill_color=RdYlGn[11][7]) # fill_alpha=0.18,



# p2.add_layout(verylow_box)

p2.add_layout(low_box)

# p2.add_layout(medium_box)

p2.add_layout(high_box)



# Adding spans to the figure

span_2 = Span(location=2, dimension='width', line_color='green', line_dash=[3,3])

# span_5 = Span(location=5, dimension='width', line_color='green', line_dash=[3,3])

span_10 = Span(location=10, dimension='width', line_color='tomato', line_dash=[3,3])

# Adding annotations to the spans

mytext_span2 = Label(x=0, y=2, text='2%', text_color = "black", text_font_size='8pt')

# mytext_span5 = Label(x=0, y=5, text='5%', text_color = "black", text_font_size='8pt')

mytext_span10 = Label(x=0, y=10, text='10%', text_color = "black", text_font_size='8pt')

p2.add_layout(span_2)

# p2.add_layout(span_5)

p2.add_layout(span_10)

p2.add_layout(mytext_span2)

# p2.add_layout(mytext_span5)

p2.add_layout(mytext_span10)



# Stylize the figure title

p2.title.text_color = "black"

p2.title.text_font = "times"

p2.title.text_font_size = "20px" # px stands for pixel

p2.title.align = "center"



# Stylize the tools

# Adding customization to HoverTool

p2.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),("Total Confirmed", "@CumConfirmed"),("Total Deaths", "@CumDeaths"),("Fatality Rate", "@FatalityRate{0.00}%")])



p2.add_tools(hover) # Customized HoverTool



# Stylize Legends

legend = (Legend(items=[("Germany", [line1]),

                        ("Korea, South", [line2]),

                        ("Finland", [line3]),

                        ("Denmark", [line4]),

                        ("Netherlands", [line5]),

                        ("Australia", [line6, circle6]),

                        ("Croatia", [line7]),

                        ("Japan", [line8]),

                        ("Ireland", [line9]),

                        ("Belarus", [line10])

                       ]))

legend.background_fill_color = "#1f77b4"

legend.background_fill_alpha = 0

legend.border_line_color = None

legend.click_policy="hide"  # To disable/hide the legend on click

p2.add_layout(legend, "right")



# 4. 

# Adding Annotations to the plot

mytext1 = Label(x=195, y=2.5, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

p2.add_layout(mytext1)



# 5. 

# Adding title subtitle and footer as a <div> tag (a division or a section in an HTML document)



title = Div(text = "<b> Fatality Rate trajectories for Top 10 countries with the highest Health Capacity Scores </b>", 

            width=700, style={'font-size': '125%', 'color': 'black'}, align = "start")

# subtitle1 = Div(text = "Health Capacity Score is a highly relevant attribute of the comprehensive Global Health Security Index relating to country's preparedness for COVID-19 <i>(https://www.ghsindex.org)</i>", 

#             width=620, style={'font-size': '100%', 'color': 'black'}, align = "start") 

subtitle2 = Div(text = "<i> Fatality Rate = (Cumulative Deaths / Cumulative Confirmed)*100 </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "start") 

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 





# 6. 

# Creating the layout

layout=column(title, subtitle2, p2, footer)#subtitle1,



# 7. 

# Displaying the layout

show(layout) 

# 1.

# Creating figure for the plot



p2 = figure(x_axis_label ='Days', y_axis_label = 'Fatality Rate (%)', y_axis_type ="linear")

#             , plot_width=850, plot_height = 400)#creating figure object 



# 2.

# Creating the graph on figure

line1 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", line_width = 2) # plotting the data using glyph circle

line2 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", line_width = 2) # plotting the data using glyph circle

line3 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", line_width = 2) # plotting the data using glyph circle

line4 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", line_width = 2) # plotting the data using glyph circle

line5 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", line_width = 2) # plotting the data using glyph circle

line6 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle

line7 = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle

line = p2.line(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle

circle = p2.circle(x='Days', y="FatalityRate", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 3) # plotting the data using glyph circle



# 3.

# Stylizing



# Stylize the plot area

p2.ygrid.grid_line_alpha = 0.7

p2.plot_width = 600

p2.plot_height = 350



# Axes Geometry

p2.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

p2.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

# p2.y_range.start = 0

# p2.x_range.start = 0



# Stylize the axes

p2.axis.axis_label_text_font_style = "bold" 

p2.axis.axis_label_text_color = "black"

p2.yaxis.minor_tick_in = 0

p2.yaxis.minor_tick_out = 0

p2.xaxis.minor_tick_in = 0

p2.xaxis.minor_tick_out = 0



# Box Annotations

verylow_box = BoxAnnotation(bottom = 0, top= 2, fill_color=RdYlGn[11][7])

low_box = BoxAnnotation(bottom=2, top=5, fill_color=RdYlGn[11][4])

medium_box = BoxAnnotation(bottom=5, top=10, fill_color=RdYlGn[11][4])

high_box = BoxAnnotation(bottom=10, fill_color=RdYlGn[11][3]) # fill_alpha=0.18,



p2.add_layout(verylow_box)

p2.add_layout(low_box)

p2.add_layout(medium_box)

p2.add_layout(high_box)



# Adding spans to the figure

span_25 = Span(location=2, dimension='width', line_color='tomato', line_dash=[3,3])

span_75 = Span(location=10, dimension='width', line_color='green', line_dash=[3,3])

# Adding Annotations to spans

mytext_span25 = Label(x=0, y=2, text='2%', text_color = "black", text_font_size='8pt')

mytext_span75 = Label(x=0, y=10, text='10%', text_color = "black", text_font_size='8pt')

p2.add_layout(span_25)

p2.add_layout(span_75)

p2.add_layout(mytext_span25)

p2.add_layout(mytext_span75)



# Adding Annotations to the plot

mytext1 = Label(x=195, y=2.5, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

p2.add_layout(mytext1)



# Stylize the figure title

p2.title.text_color = "black"

p2.title.text_font = "times"

p2.title.text_font_size = "20px" # px stands for pixel. Have to mention.

p2.title.align = "center"



# Stylize the tools

# Adding customization to HoverTool

p2.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),("Total Confirmed", "@CumConfirmed"),("Total Recovered", "@CumRecovered"),("Recovery Rate", "@RecoveryRate{0.00}")])



p2.add_tools(hover) # Customization of HoverTool



# Stylize Legends

legend = (Legend(items=[("China", [line1]),

                        ("India", [line2]),

                        ("US", [line3]),

                        ("Indonesia", [line4]),

                        ("Pakistan", [line5]),

                        ("Brazil", [line6]),

                        ("Nigeria", [line7]),

                        ("Australia", [line, circle])

                        ]))

legend.background_fill_color = "#1f77b4"

legend.background_fill_alpha = 0

legend.border_line_color = None

legend.click_policy="hide"  # To disable/hide the legend on click

p2.add_layout(legend, "right")



# 4.

# Adding title subtitle and footer as a <div> tag (a division or a section in an HTML document)



title = Div(text = "<b> Comparison of Fatality Rate trajectories for <br>Top 7 highly populated countries and moderately populated Australia </b>", 

            width=600, style={'font-size': '125%', 'color': 'black'}, align = "start")

subtitle = Div(text = "<i> Fatality Rate = Cumulative Deaths / Cumulative Confirmed </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "start") 

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 



# 5.

# Creating the layout

layout1=column(title, subtitle, p2, footer)



# 6.

# Displaying the layout

show(layout1) 

# Step 1:

# Creating ColumnDataSources 



data_all=data.loc[:, ['Country','Days', 

                      'Daily_Confirmed_7day_rolling_average_per_million', "CumConfirmed", 

                      'Daily_testing_7day_rolling_average_per_thousand', "CumTesting",

                      "HealthCapacityScore", "Population"]]



Overall = ColumnDataSource(data=data_all)

Curr = ColumnDataSource(dict(Country = [], Days = [], 

                             Daily_Confirmed_7day_rolling_average_per_million = [], CumConfirmed = [],

                             Daily_testing_7day_rolling_average_per_thousand = [], CumTesting = [],

                             HealthCapacityScore = [], Population = []))



#------------------------------------------------------------------------------------------

# Step 2:

# Defining callback function which links plots and the select menu



multi_countries_callback = CustomJS(args=dict(source=Overall, current=Curr), code="""

var Selected_Country = cb_obj.value



current.data['Country'] = []

current.data['Days']=[]

current.data['Daily_Confirmed_7day_rolling_average_per_million'] = []

current.data['CumConfirmed'] = []

current.data['CumTesting'] = []

current.data['Daily_testing_7day_rolling_average_per_thousand'] = []

current.data['HealthCapacityScore'] = []

current.data['Population'] = []



for(var i = 0; i <= source.get_length(); i++){

	if (Selected_Country.indexOf(source.data['Country'][i]) >= 0){

        current.data['Country'].push(source.data['Country'][i])

		current.data['Days'].push(source.data['Days'][i])

        current.data['Daily_Confirmed_7day_rolling_average_per_million'].push(source.data['Daily_Confirmed_7day_rolling_average_per_million'][i])

        current.data['CumConfirmed'].push(source.data['CumConfirmed'][i])

        current.data['CumTesting'].push(source.data['CumTesting'][i])

        current.data['Daily_testing_7day_rolling_average_per_thousand'].push(source.data['Daily_testing_7day_rolling_average_per_thousand'][i])

        current.data['HealthCapacityScore'].push(source.data['HealthCapacityScore'][i])

        current.data['Population'].push(source.data['Population'][i])

    }          

} 

current.change.emit();

""")



#-------------------------------------------------------------------------------------------

# Step 3: Creating menu

menu = MultiSelect(options=list(data['Country'].unique()),value=[], title= "Select one or more Country/Region:" ) # drop down menu



#------------------------------------------------------------------------------------------

# Step 4: Creating plots



# A. 

#------------------------------------------------------------------------------------------

# Plot for 7-day rolling average of Count of Confirmed Cases (per million) 

#------------------------------------------------------------------------------------------



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P = [] 

axis_type =["linear", "log"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Count", y_axis_type =axis_type[i])#creating figure object 

    p3.add_layout(Title(text="Trajectories for 7 day average of count of confirmed cases (per million)", text_font_size="10pt"), 'above')



# 3.

# Creating the graph on figure

    line1 = p3.line(x='Days', y= 'Daily_Confirmed_7day_rolling_average_per_million', source=Curr, color = "tomato", line_width = 2) 

    circle1 = p3.circle(x='Days', y= 'Daily_Confirmed_7day_rolling_average_per_million', source=Curr, color = "tomato", size = 2) 



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 550                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

    p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(end = 450)



    # Stylize the axes

    # p3.yaxis.formatter.use_scientific = False

    # p3.xaxis.formatter.use_scientific = False

    p3.axis.axis_label_text_font_style = "bold" 

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customization to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("7 day average for confirmed cases per million", "@Daily_Confirmed_7day_rolling_average_per_million"),

                                  ("7 day average for testing per thousand", "@Daily_testing_7day_rolling_average_per_thousand{int}"), 

                                  ("Cumulative Confirmed", "@CumConfirmed"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("Population", "@Population{0.00}")

                                  ])



    p3.add_tools(hover) # Customized HoverTool



    

# 5.

# Appending the figures for log and linear scale in list P

    P.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Linear scale")

tab2 = Panel(child=P[1], title="Log scale")



# 7.

# Creating list of Tabs

tabs_confirmed = Tabs(tabs=[ tab1, tab2 ])





# B. 

#------------------------------------------------------------------------------------------

# Plot for 7-day average of Testing Count (per thousand) for all countries individually

#------------------------------------------------------------------------------------------



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P = [] 

axis_type =["linear", "log"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Count", y_axis_type =axis_type[i])#creating figure object 

    p3.add_layout(Title(text="Trajectories for 7 day average of testing count (per thousand)", text_font_size="10pt"), 'above')



# 3.

# Creating the graph on figure

    line2 = p3.line(x='Days', y= 'Daily_testing_7day_rolling_average_per_thousand', source=Curr, color = "navy", line_width = 2) # plotting the data using glyph circle

    circle2 = p3.circle(x='Days', y= 'Daily_testing_7day_rolling_average_per_thousand', source=Curr, color = "navy", size = 2)



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 550                

    p3.plot_height = 350                 

    p3.background_fill_color = "#1f77b4"   

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

    p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(end = 10)

#     if axis_type[i] == "linear":

#         p3.yaxis.formatter.use_scientific = False



    # Stylize the axes

    # p3.yaxis.formatter.use_scientific = False

    # p3.xaxis.formatter.use_scientific = False

    p3.axis.axis_label_text_font_style = "bold" 

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customization to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("7 day average for testing per thousand", "@Daily_testing_7day_rolling_average_per_thousand{int}"),

                                  ("7 day average for confirmed cases per million", "@Daily_Confirmed_7day_rolling_average_per_million"),

                                  ("Cumulative Confirmed", "@CumConfirmed"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("Population", "@Population{0.00}")

                                ])



    p3.add_tools(hover) # Customized HoverTool



    

# 5.

# Appending the figures for log and linear y-axis scale

    P.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Linear scale")

tab2 = Panel(child=P[1], title="Log scale")



# 7.

# Creating list of Tabs

tabs_testing = Tabs(tabs=[ tab1, tab2 ])



#------------------------------------------------------------------------------------------

# Step 5: Calling the function on change of selection



menu.js_on_change('value', multi_countries_callback) 



#------------------------------------------------------------------------------------------

# Step 6: Adding common footers to both plots as <div> tags (a division or a section in an HTML document)



footer1 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

footer2 = Div(text = "<i> Testing Rate this day = Confirmed this day / Tested this day </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 

footer3 = Div(text = "<i> Cumulative Testing Rate = Cumulative Testing / Cumulative Confirmed </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 



#------------------------------------------------------------------------------------------

# Step 7: Creating the layout



layout3=column(menu, tabs_confirmed, tabs_testing , footer1, footer2, footer3)



#------------------------------------------------------------------------------------------

# Step 8: Displaying the layout

show(layout3) 
# df_hcs.sort_values('Score/100',ascending = False).head(8)
# Creating plots



# A. 

#------------------------------------------------------------------------------------------

# 7-day rolling average of Count of Confirmed Cases (per million)

#------------------------------------------------------------------------------------------



# 1. 

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P1 = [] 

axis_type =["linear", "log"]



# 2.

# Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Count", y_axis_type =axis_type[i])

#     , plot_width=850, plot_height=400)#creating figure object 





# 3.

# Creating the graph on figure



    circle1 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", line_width = 2) # plotting the data using glyph circle

    

    circle2 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle



    circle3 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", line_width = 2) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", line_width = 2) # plotting the data using glyph circle

    

    circle5 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", size =2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", line_width = 2) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 2.5) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 3) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle



    circle8 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", size = 2) # plotting the data using glyph circle

    line8 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", line_width = 2) # plotting the data using glyph circle



    circle9 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Ireland"]), color = "mediumvioletred", size = 2) # plotting the data using glyph circle

    line9 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Ireland"]), color = "mediumvioletred", line_width = 2) # plotting the data using glyph circle

    

    circle10 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Belarus"]), color = "gold", size = 2) # plotting the data using glyph circle

    line10 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Belarus"]), color = "gold", line_width = 2) # plotting the data using glyph circle



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

#     p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(end = 250)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

    

    # Stylize the axes  

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customizayion to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("7 day average for confirmed cases per million", "@Daily_Confirmed_7day_rolling_average_per_million{int}"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("7 day average for testing per thousand", "@Daily_testing_7day_rolling_average_per_thousand{int}"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}"),

                                 ])



    p3.add_tools(hover) # Customization of HoverTool

    

    # Adding Annotations

    mytext1 = Label(x=179, y=22, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)



    # Stylize Legends

    legend = (Legend(items=[("Germany", [circle1, line1]),

                            ("Korea, South", [circle2, line2]),

                            ("Finland", [circle3, line3]),

                            ("Denmark", [circle4, line4]),

                            ("Netherlands", [circle5, line5]),

                            ("Australia", [circle6, line6]),

                            ("Croatia", [circle7, line7]),

                            ("Japan", [circle8, line8]),

                            ("Ireland", [circle9, line9]),

                            ("Belarus", [circle10, line10])

                            ]))

    

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")

# 5.

# Appending the figures for log and linear scale of y-axis

    P1.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab11 = Panel(child=P1[0], title="Linear scale")

tab12 = Panel(child=P1[1], title="Log scale")



# 7.

# Creating list of Tabs

tabs1 = Tabs(tabs=[tab11, tab12])



# 8.

# Adding title and footer as a <div> tag (a division or a section in an HTML document)

title1 = Div(text = "<b> Trajectories for <i> 7 day average count for Confirmed Cases (per million)</i> <br> for Top 10 countries with highest Health Capacity Scores </b>", 

           style={'font-size': '125%', 'color': 'black'}, align = "start")

footer1 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

# 9. 

# Creating layout

layout1 = column(title1, tabs1, footer1)







# B) 

#------------------------------------------------------------------------------------------

# 7-day average of Testing Count (per thousand) 

#------------------------------------------------------------------------------------------



# 1. 



# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P2 = [] 

axis_type =["linear", "log"]



# 2. Creating figure for the plot

for i in range(2):

    p3 = figure(x_axis_label ='Days', y_axis_label = "Count", y_axis_type =axis_type[i])

#     , plot_width=850, plot_height=400)#creating figure object 





# 3. Creating the graph on figure 



    circle1 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Germany"]), color = "darkgreen", line_width = 2) # plotting the data using glyph circle

    

    circle2 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Korea, South"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle



    circle3 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Finland"]), color = "purple", line_width = 2) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Denmark"]), color = "maroon", line_width = 2) # plotting the data using glyph circle



    circle5 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", size = 2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Netherlands"]), color = "crimson", line_width = 2) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 2) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 2.5) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Croatia"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle



    circle8 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", size = 2) # plotting the data using glyph circle

    line8 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Japan"]), color = "orangered", line_width = 2) # plotting the data using glyph circle



    circle9 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Ireland"]), color = "mediumvioletred", size = 2) # plotting the data using glyph circle

    line9 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Ireland"]), color = "mediumvioletred", line_width = 2) # plotting the data using glyph circle

    

    circle10 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Belarus"]), color = "gold", size = 2) # plotting the data using glyph circle

    line10 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Belarus"]), color = "gold", line_width = 2) # plotting the data using glyph circle



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

#     p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(end = 10)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

    

    # Stylize the axes  

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customizayion to HoverTool

#     p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("7 day average for testing per thousand", "@Daily_testing_7day_rolling_average_per_thousand{int}"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("7 day average for confirmed cases per million", "@Daily_Confirmed_7day_rolling_average_per_million{int}"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}"),

                                 ])



    p3.add_tools(hover) # Customization of HoverTool

    

    # Adding Annotations

    mytext1 = Label(x=182, y=3, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)



    #Stylize Legends

    legend = (Legend(items=[("Germany", [circle1, line1]),

                            ("Korea, South", [circle2, line2]),

                            ("Finland", [circle3, line3]),

                            ("Denmark", [circle4, line4]),

                            ("Netherlands", [circle5, line5]),

                            ("Australia", [circle6, line6]),

                            ("Croatia", [circle7, line7]),

                            ("Japan", [circle8, line8]),

                            ("Ireland", [circle9, line9]),

                            ("Belarus", [circle10, line10])

                        ]))

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")



# 5. 

# Appending the figures for log and linear scale for y-axis 

    P2.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab21 = Panel(child=P2[0], title="Linear scale")

tab22 = Panel(child=P2[1], title="Log scale")



# 7. 

# Creating list of Tabs

tabs2 = Tabs(tabs=[tab21, tab22])



# 8.

# Adding title and footer as a <div> tag (a division or a section in an HTML document)



title2 = Div(text = "<b> Trajectories for <i>7 day average for Testing Count (per thousand)</i> <br> for Top 10 countries with highest Health Capacity Scores </b>", 

            style={'font-size': '125%', 'color': 'black'}, align = "start")



footer2 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

# 9. 

# Creating layout



layout2 = column(title2, tabs2, footer2)

#------------------------------------------------------------------------------------------

# Adding common footers for both plots as <div> tags (a division or a section in an HTML document)



footer = Div(text = "Health Capacity Score is a highly relevant attribute of the comprehensive Global Health Security Index relating to country's <br> preparedness for COVID-19 <i>(https://www.ghsindex.org)</i>", 

            style={'font-size': '100%', 'color': 'black'}, align = "center") 

#------------------------------------------------------------------------------------------

# Creating and showing the layout

layout = column(layout1, layout2, footer)

show(layout)
# Creating plots



# A)

#------------------------------------------------------------------------------------------

# Daily Confirmed Count plot for top 5 populated countries and Australia

#------------------------------------------------------------------------------------------



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot

P = [] 

axis_type =["linear", "log"]



# 2.

# Creating figure for the plot

for i in range(2):

 

    p3 = figure(x_axis_label ='Days', y_axis_label = "Count", y_axis_type =axis_type[i])

#     , plot_width=850, plot_height=400)#creating figure object 



# 3.

# Creating the graph on figure



    circle1 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", line_width = 2) # plotting the data using glyph circle

    

    circle2 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", line_width = 2) # plotting the data using glyph circle



    circle3 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", line_width = 2) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", line_width = 2) # plotting the data using glyph circle



    

    circle5 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", size =2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", line_width = 2) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", size =2) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle



    circle = p3.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 3) # plotting the data using glyph circle

    line = p3.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 2) # plotting the data using glyph circle





# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

#     p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(end = 250)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False



    # Stylize the axes

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customization to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("7 day average for confirmed cases per million", "@Daily_Confirmed_7day_rolling_average_per_million{int}"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("7 day average for testing per thousand", "@Daily_testing_7day_rolling_average_per_thousand{int}"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}"),

                                 ])



    p3.add_tools(hover) # Customized HoverTool



     # Adding Annotations

    mytext1 = Label(x=175, y=19.8, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)

    

    # Stylize Legends

    legend = (Legend(items=[("China", [circle1, line1]),

                            ("India", [circle2, line2]),

                            ("US", [circle3, line3]),

                            ("Indonesia", [circle4, line4]),

                            ("Pakistan", [circle5, line5]),

                            ("Brazil", [circle6, line6]),

                            ("Nigeria", [circle7, line7]),

                            ("Australia", [circle, line])

                            ]))

    

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")



# 5.

# Appending the figures with log and linear scale for y-axis

    P.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Linear scale")

tab2 = Panel(child=P[1], title="Log scale")



# 7. 

# Creating list of Tabs

tabs_confirmed = Tabs(tabs=[ tab1, tab2 ])



# 8. 

# Adding title and a footer as a <div> tag (a division or a section in an HTML document)



title1 = Div(text = "<b>Comparison of trajectories of <i>7-day average Count of Confirmed Cases (per million)</i> of <br>Top 7 Populated countries and Australia</b>", 

            style={'font-size': '125%', 'color': 'black'}, align = "start") # width=600, 



footer1 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")





# 9.

# Creating the layout

layout1=column(title1, tabs_confirmed, footer1)



#------------------------------------------------------------------------------------------





# B)

#------------------------------------------------------------------------------------------

# Daily testing Count plot for top 7 populated countries and Australia

#------------------------------------------------------------------------------------------



# 1. 

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot



P = [] 

axis_type =["linear", "log"]



# 2.

# Creating figure for the plot

for i in range(2):

    

    p3 = figure(x_axis_label ='Days', y_axis_label = "Count", y_axis_type =axis_type[i])

#     , plot_width=850, plot_height=400)#creating figure object 





# 3.

# Creating the graph on figure



    circle1 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", size = 2) # plotting the data using glyph circle

    line1 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="China"]), color = "darkgreen", line_width = 2) # plotting the data using glyph circle

    

    circle2 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", size = 2) # plotting the data using glyph circle

    line2 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="India"]), color = "orange", line_width = 2) # plotting the data using glyph circle



    circle3 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", size = 2) # plotting the data using glyph circle

    line3 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="US"]), color = "red", line_width = 2) # plotting the data using glyph circle



    circle4 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", size = 2) # plotting the data using glyph circle

    line4 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Indonesia"]), color = "purple", line_width = 2) # plotting the data using glyph circle



    

    circle5 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", size =2) # plotting the data using glyph circle

    line5 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Pakistan"]), color = "maroon", line_width = 2) # plotting the data using glyph circle



    circle6 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", size =2) # plotting the data using glyph circle

    line6 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Brazil"]), color = "dodgerblue", line_width = 2) # plotting the data using glyph circle



    circle7 = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", size = 2) # plotting the data using glyph circle

    line7 = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Nigeria"]), color = "slateblue", line_width = 2) # plotting the data using glyph circle



    circle = p3.circle(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", size = 3) # plotting the data using glyph circle

    line = p3.line(x='Days', y="Daily_testing_7day_rolling_average_per_thousand", source=ColumnDataSource(data[data["Country"]=="Australia"]), color = "navy", line_width = 2) # plotting the data using glyph circle



# 4.

# Stylizing



    # Stylize the plot area

    p3.plot_width = 600                # To change the width of the plot

    p3.plot_height = 350                 # To change the height of plot

    p3.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p3.background_fill_alpha = 0.12



    # Stylize the grid

    p3.xgrid.grid_line_color = "white"

    p3.ygrid.grid_line_color = "white"

    p3.ygrid.grid_line_alpha = 0.7

    p3.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p3.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p3.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

#     p3.y_range.start = 0

    if axis_type[i] == "log":

        p3.y_range = DataRange1d(end = 4)

    if axis_type[i] == "linear":

        p3.yaxis.formatter.use_scientific = False

    

    # Stylize the axes

    p3.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p3.axis.axis_label_text_color = "black"

    # p3.axis.major_label_text_font_style = "bold" # for axis' major tick marks' labels value color

    p3.yaxis.minor_tick_in = 0

    p3.yaxis.minor_tick_out = 0

    p3.xaxis.minor_tick_in = 0

    p3.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p3.title.text_color = "black"

    p3.title.text_font = "times"

    p3.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p3.title.align = "center"



    # Stylize the tools

    # Adding customizayion to HoverTool

    p3.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Day", "@Days"),

                                  ("7 day average for testing per thousand", "@Daily_testing_7day_rolling_average_per_thousand{int}"),

                                  ("Cumulative Testing Count", "@CumTesting"),

                                  ("7 day average for confirmed cases per million", "@Daily_Confirmed_7day_rolling_average_per_million{int}"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed"),

                                  ("Health Capacity Score", "@HealthCapacityScore{0.00}"),

                                  ("Population", "@Population{0.00}"),

                                 ])

    p3.add_tools(hover) # Customization of HoverTool

    

    # Adding Annotations

    mytext1 = Label(x=175, y=3, text='Australia', text_color = "navy", text_font_size='8pt', text_font_style = "bold")

    p3.add_layout(mytext1)



    # Stylize Legends

    legend = (Legend(items=[("China", [circle1, line1]),

                            ("India", [circle2, line2]),

                            ("US", [circle3, line3]),

                            ("Indonesia", [circle4, line4]),

                            ("Pakistan", [circle5, line5]),

                            ("Brazil", [circle6, line6]),

                            ("Nigeria", [circle7, line7]),

                            ("Australia", [circle, line])

                            ]))

        

    legend.background_fill_color = "#1f77b4"

    legend.background_fill_alpha = 0

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p3.add_layout(legend, "right")



# 5.

# Appending the figures with log and linear scale for y-axis

    P.append(p3)

    

# 6.

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Linear scale")

tab2 = Panel(child=P[1], title="Log scale")



# 7. 

# Creating list of Tabs

tabs_testing = Tabs(tabs=[ tab1, tab2 ])





# 8.

# Adding title and footer as a <div> tag (a division or a section in an HTML document)



title2 = Div(text = "<b> Comparison of trajectories for <i>7-day average Testing Count (per thousand)</i> of <br>Top 7 Populated countries and moderately populated Australia</b>", 

           style={'font-size': '125%', 'color': 'black'}, align = "start") # width=600, 



footer2 = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q></i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center")

# 9. 

# Creating the layout

layout2=column(title2, tabs_testing, footer2)



#------------------------------------------------------------------------------------------

# Creating the final layout and displaying

layout = column(layout1, layout2)

show(layout) 

countries = ["Afghanistan", "Australia", "Austria", "Belarus", "Belgium", "Canada", "Chad", "China", "Croatia", "Cuba", 

             "Cyprus", "Denmark", "Djibouti", "Estonia", "Finland", "France", "Germany", "Hungary", "Iceland", "Ireland", 

             "Italy","Japan", "Korea, South", "Latvia", "Luxembourg", "Netherlands", "New Zealand", "Norway", "Qatar",

             "Singapore","Slovakia", "Slovenia", "Switzerland", "Taiwan", "Tajikistan","Thailand", "Tunisia", "Turkey",

             "United Arab Emirates", "United Kingdom"]

#-------------------------------------------------------------------------------------------



# Creating plot

# Creating an empty list for figures for plots created  for above set of countries



P = []



for country in countries:

    

# Step 1 Creating figure for the plot    

    p7 = figure( y_axis_label = 'Count', x_axis_label ='Days', y_axis_type = "linear")#creating figure object 

    p7.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=ColumnDataSource(data[data["Country"]==country]), color = "black", line_width = 2, legend_label = country) # plotting the data using glyph circle



    # Stylize the plot area

    p7.plot_width = 200                # To change the width of the plot

    p7.plot_height = 250                 # To change the height of plot

    p7.background_fill_color = "#1f77b4"   # To add background colorto the figure

    p7.background_fill_alpha = 0.12



    # Stylize the grid

    p7.xgrid.grid_line_color = "white"

    p7.ygrid.grid_line_color = "white"

    p7.ygrid.grid_line_alpha = 0.7

    p7.grid.grid_line_dash = [5,3]



    # Axes Geometry

    p7.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p7.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends



    # Stylize the axes

    p7.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p7.axis.axis_label_text_color = "black"

    # p.axis.major_label_text_font_style = "bold" # for axis' major tick marks' labels value color

    p7.yaxis.minor_tick_in = 0

    p7.yaxis.minor_tick_out = 0

    p7.xaxis.minor_tick_in = 0

    p7.xaxis.minor_tick_out = 0

    p7.xaxis.ticker.desired_num_ticks = 4



    # Stylize the tools

    # Adding customizayion to HoverTool

    p7.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [(("Day", "@Days")), ("7-day avg count of Confirmed Cases(per million)", "@Daily_Confirmed_7day_rolling_average_per_million"),

                                  ("Cumulative Confirmed Count", "@CumConfirmed")])

    p7.add_tools(hover) # Customization of HoverTool

   

    # Stylize the legends

    p7.legend.background_fill_alpha = 0

    p7.legend.border_line_color = None

    p7.legend.location ="top_right"

    p7.legend.label_text_font_size = '12px'

    p7.legend.label_text_font = "times"

    P.append(p7)

   

    #------------------------------------------------------------------------------------------



# Creating the layout



f = gridplot([ [P[0], P[1], P[2]],

               [P[3], P[4], P[5]],

               [P[6], P[7], P[8]],

               [P[9], P[10], P[11]],

               [P[12], P[13], P[14]],

               [P[15], P[16], P[17]],

               [P[18], P[19], P[20]],

               [P[21], P[22], P[23]],

               [P[24], P[25], P[26]],

               [P[27], P[28], P[29]],

               [P[30], P[31], P[32]],

               [P[33], P[34], P[35]],

               [P[36], P[37], P[38]],

               [P[39]]

               ])



# f = gridplot([ [P[0], P[1], P[2],

#                 P[3], P[4], P[5]],

#                [P[6], P[7], P[8],

#                P[9], P[10], P[11]],

#                [P[12], P[13], P[14],

#                P[15], P[16], P[17]],

#                [P[18], P[19], P[20],

#                P[21], P[22], P[23]],

#                [P[24], P[25], P[26],

#                P[27], P[28], P[29] ],

#                [P[30], P[31],P[32],

#                 P[33], P[34], P[35]],

#                [P[36], P[37],P[38],

#                 P[39]]

#                ])



# Defining the title and subtitles for complete gridplot as <div> tags (a division or a section in an HTML document)

title = Div(text = "<b> Trajectory of 7-day avg count of Confirmed Cases(per million) <br> for the countries which have flattened the curve </b>", 

            width=900, style={'font-size': '150%', 'color': 'black'}, align = "center") # height = 50,

subtitle = Div(text = "<b><i> (Day 1 is the day on which the country encountered its first 10 total confirmed cases) </i></b>", 

            width=900, style={'font-size': '100%', 'color': 'black'}, align = "center") # height = 50,

#------------------------------------------------------------------------------------------

# Displaying the layout

show(column(title,subtitle, f))

# Step 1: Creating ColumnDataSources 



data_all=data.loc[:, ['Country','Days', 'CumConfirmed','DailyConfirmed', 'Daily_Confirmed_7day_rolling_average_per_million',

                      'CumDeaths', 'CumRecovered',

                      'CumTesting', 'smoothed_cumulative_testing_count']]

data_curr = data_all[data_all['Country'] == 'Australia' ]



Overall = ColumnDataSource(data=data_all)

Curr=ColumnDataSource(data=data_curr)



#------------------------------------------------------------------------------------------



# Step 2:

# Defining callback function which links plots and the select menu



callback = CustomJS(args=dict(source=Overall, current=Curr), code="""

var selected_country = cb_obj.value

current.data['Days']=[]

current.data['CumConfirmed']=[]

current.data['DailyConfirmed']=[]

current.data['Daily_Confirmed_7day_rolling_average_per_million'] = []

current.data['CumDeaths'] = []

current.data['CumRecovered'] = []

current.data['CumTesting'] = []

current.data["smoothed_cumulative_testing_count"] = []

for(var i = 0; i <= source.get_length(); i++){

	if (source.data['Country'][i] == selected_country){

		current.data['Days'].push(source.data['Days'][i])

		current.data['CumConfirmed'].push(source.data['CumConfirmed'][i])

        current.data['DailyConfirmed'].push(source.data['DailyConfirmed'][i])

		current.data['CumDeaths'].push(source.data['CumDeaths'][i])

		current.data['CumRecovered'].push(source.data['CumRecovered'][i])

		current.data['CumTesting'].push(source.data['CumTesting'][i]) 

        current.data["smoothed_cumulative_testing_count"].push(source.data["smoothed_cumulative_testing_count"][i])

        current.data['Daily_Confirmed_7day_rolling_average_per_million'].push(source.data['Daily_Confirmed_7day_rolling_average_per_million'][i])

     }          

} 

current.change.emit();

""")



#-------------------------------------------------------------------------------------------

# Step 3: Creating menu

menu = Select(options=list(data['Country'].unique()),value='Australia', title="Select Country:")  # drop down menu



#------------------------------------------------------------------------------------------

# Step 4: Creating plot



# 1.

# Creating empty list for storing two figures :1 with LogScale for y-axis and other with LinearScale

# These two plots are shown in separate tabs in he same plot

P = [] 

axis_type =["linear", "log"]



# 2.

# Creating figure for the plot

for i in range(2):

    p1 = figure(x_axis_label ='Days', y_axis_label = 'Count', y_axis_type = axis_type[i])#creating figure object 



# 3.

# Plotting line graph on the figure

       

    line1 = p1.line(x='Days', y="DailyConfirmed", source=Curr, color = "black", line_width = 2) # plotting the data using glyph circle

    circle1 = p1.circle(x='Days', y="DailyConfirmed", source=Curr, color = "black", size = 1)

    

#     line1 = p1.line(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=Curr, color = "black", line_width = 2) # plotting the data using glyph circle

#     circle1 = p1.circle(x='Days', y="Daily_Confirmed_7day_rolling_average_per_million", source=Curr, color = "black", size = 1)

    

# 4.

# Stylizing



    # Stylize the plot area

    p1.plot_width = 600               

    p1.plot_height = 350                 

    p1.background_fill_color = "#1f77b4"   

    p1.background_fill_alpha = 0.12

    

    # Stylize the grid

    p1.xgrid.grid_line_color = "white"

    p1.ygrid.grid_line_color = "white"

    p1.ygrid.grid_line_alpha = 0.7

    p1.grid.grid_line_dash = [5,3]

    

    # Axes Geometry

    p1.x_range = DataRange1d(only_visible=True) # x_range changes according to the active legends

    p1.y_range = DataRange1d(only_visible=True) # y_range changes according to the active legends

    

    # Stylize the axes

    p1.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p1.axis.axis_label_text_color = "black"

    p1.yaxis.minor_tick_in = 0

    p1.yaxis.minor_tick_out = 0

    p1.xaxis.minor_tick_in = 0

    p1.xaxis.minor_tick_out = 0



    # Stylize the figure title

    p1.title.text_color = "black"

    p1.title.text_font = "times"

    p1.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p1.title.align = "center"



    # Stylize the tools

    # Adding customization to HoverTool

    p1.tools = [ZoomInTool(),ZoomOutTool(), ResetTool()]

    hover = HoverTool(tooltips = [(("Day", "@Days")), ("Daily Confirmed", "@DailyConfirmed"), ("Cumulative Confirmed Count", "@CumConfirmed"), ("Cumulative Death Count", "@CumDeaths"), ("Cumulative Recovered Count", "@CumRecovered"), ("Cumulative Testing Count", "@smoothed_cumulative_testing_count")])

#     ("7-day average count for confirmed cases (per million)", "@Daily_Confirmed_7day_rolling_average_per_million")

    p1.add_tools(hover) # Customized HoverTool

           

    # Appending the two figures created under for loop (one with y_axis_type="log" and other

    # with y_axis_type = "linear") into the list P

    P.append(p1)



# 5. 

# Creating separate tabs for the two figures created above

tab1 = Panel(child=P[0], title="Linear scale")

tab2 = Panel(child=P[1], title="Log scale")



#------------------------------------------------------------------------------------------

# Step 5: Creating list of Tabs

tabs = Tabs(tabs=[ tab1, tab2 ])



#------------------------------------------------------------------------------------------

# Step 6: Calling the function on change of selection



menu.js_on_change('value', callback) 



#------------------------------------------------------------------------------------------

# Step 7: Adding title and footer as a <div> tag (a division or a section in an HTML document)



title = Div(text = "<b> Daily Confirmed count of Confirmed Cases for individual countries </b>", 

            width=600, style={'font-size': '125%', 'color': 'black'}, align = "center")

# title = Div(text = "<b> 7-day average count of Confirmed Cases(per million) for individual countries </b>", 

#             width=600, style={'font-size': '125%', 'color': 'black'}, align = "center")

footer = Div(text = "<i> Day 1 is the day on which the country reached 10 <q>Total Confirmed Cases</q> </i>", 

            width=600, style={'font-size': '100%', 'color': 'black'}, align = "center") 



#------------------------------------------------------------------------------------------

# Step 8: Creating the layout



layout=column(title,menu,tabs, footer) 



#------------------------------------------------------------------------------------------

# Step 9: Displaying the layout

show(layout) 

# Filtering only cumulative count columns from data and leaving daily or 7-day avaerage columns

data1 = data[["Country", "Days", 

              "CumConfirmed", "CumConfirmed_per_million", 

              "CumDeaths",

              "CumRecovered", 

              "CumTesting","smoothed_cumulative_testing_count",

              "Population", "Pop_Density", "Land_Area_Kmsq", "HealthCapacityScore", "RecoveryRate", "FatalityRate","Active", 

              "SmoothedCumTesting_per_thousand","SmoothedCumTesting_per_CumConfirmed",

              "CumConfirmed_per_SmoothedCumTesting_percent"

              ]]

# data1["Country"].unique()
# data2
data2 = data1.loc[data1["Country"].isin(["Germany","Korea, South","Finland","Denmark","Netherlands","Australia", "Croatia",

                                         "Japan","Ireland","Belarus"])]

# Creating empty dataframe to include only latest data

data2_tails = pd.DataFrame(columns=["Country", "Days", 

              "CumConfirmed", "CumConfirmed_per_million", 

              "CumDeaths",

              "CumRecovered", 

              "CumTesting","smoothed_cumulative_testing_count",

              "Population", "Pop_Density", "Land_Area_Kmsq", "HealthCapacityScore", "RecoveryRate", "FatalityRate","Active", 

              "SmoothedCumTesting_per_thousand",

              "CumConfirmed_per_SmoothedCumTesting_percent"])

# Feeding values to empty dataframe

for country in data2["Country"].unique():

#     df_tails = pd.concat([df_tails, (data1.loc[data["Country"]==country].sort_values(by=['CumTesting', 'CumConfirmed'], ascending=[False, False]).head(1))], ignore_index = True)

    data2_tails = pd.concat([data2_tails, (data2.loc[data["Country"]==country].sort_values(by=['smoothed_cumulative_testing_count', 'CumConfirmed'], ascending=[False, False]).iloc[[1]])], ignore_index = True)  
# data1.loc[data1["Country"].isin(["Germany","Korea, South","Finland","Denmark","Netherlands","Australia", "Croatia",

#                                          "Japan","Ireland","Belarus"])]
# data2_tails


# Creating the figure

p4 = figure(x_axis_label = "Cumulative Confirmed Count (per million)" , y_axis_label = 'Positive Tests Rate (%)', y_axis_type ="linear",x_axis_type ="linear") #creating figure object 

#     p4.add_layout(Title(text="Confirmed Cases Vs. ' + x + ' for the Countries', text_font_size="11pt", above, 'center'))



# Plotting the graph

# p4.circle(x="CumConfirmed", y = "CumConfirmed_per_SmoothedCumTesting_percent", size =10, fill_alpha = 0.65,line_dash = [3,3], 

#                source = ColumnDataSource(data2_tails_tails), color="navy")





circle1 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Germany"]), color = "darkgreen", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle2 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Korea, South"]), color = "dodgerblue", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle3 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Finland"]), color = "purple", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle4 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Denmark"]), color = "maroon", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle5 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Netherlands"]), color = "crimson", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle6 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Australia"]), color = "navy", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle7 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Croatia"]), color = "slateblue", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle8 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Japan"]), color = "orangered", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle9 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Ireland"]), color = "mediumvioletred", size = 10, line_dash = [3,3]) # plotting the data using glyph circle

circle10 = p4.circle(x="CumConfirmed_per_million", y="CumConfirmed_per_SmoothedCumTesting_percent", source=ColumnDataSource(data2_tails[data2_tails["Country"]=="Belarus"]), color = "gold", size = 10, line_dash = [3,3]) # plotting the data using glyph circle



p4.plot_width = 650                     # To change the width of the plot

p4.plot_height = 400                    # To change the height of plot

p4.background_fill_color = "ghostwhite"   # To add background colorto the figure

# p.background_fill_alpha = 0.12



# Stylize the title

#     p4.title.text = "Title With Options"

p4.title.align = "left"

p4.title.text_color = "red"

p4.title.text_font_size = "10px"

#     p4.title.background_fill_color = "#aaaaee"





# Stylize the grid

p4.xgrid.grid_line_color = "white"

p4.ygrid.grid_line_color = "white"

p4.ygrid.grid_line_alpha = 0.7

p4.xgrid.grid_line_dash = [5,3]

p4.xgrid.grid_line_width = 2

p4.grid.grid_line_dash = [5,3]

p4.grid.grid_line_width = 2





# Axes Geometry

# p4.x_range = DataRange1d(start = 0, end = 1000 ) # x_range changes according to the active legends

p4.y_range = DataRange1d(start = 0, only_visible=True) # y_range changes according to the active legends

p4.x_range = DataRange1d(start = 0, only_visible=True)

p4.yaxis.formatter.use_scientific = False

p4.xaxis.formatter.use_scientific = False



# Stylize the axes

#     p4.xaxis.major_label_orientation = "vertical"

p4.yaxis.ticker.desired_num_ticks = 8 

p4.axis.axis_label_text_font_style = "bold" #"normal", "italic"

p4.axis.axis_label_text_color = "black"

# p4.xaxis.major_label_orientation = "vertical"

# p.axis.major_label_text_font_style = "bold" # for axis' major tick marks' labels value color

p4.yaxis.minor_tick_in = 0

p4.yaxis.minor_tick_out = 0

p4.xaxis.minor_tick_in = 0

p4.xaxis.minor_tick_out = 0





# Stylize the figure title

p4.title.text_color = "black"

p4.title.text_font = "times"

p4.title.text_font_size = "20px" # px stands for pixel. Have to mention.

p4.title.align = "center"







#     # Stylize the tools

# Adding customization to HoverTool

p4.tools = [ZoomInTool(),ZoomOutTool(), ResetTool(), BoxZoomTool()]

hover1 = HoverTool(tooltips = [("Country", "@Country"),("Total Confirmed", "@CumConfirmed"),

                               ("Cumulative Confirmed per million", "@CumConfirmed_per_million{0.00}"),

                               ("Positive Testing Ratio", "@CumConfirmed_per_SmoothedCumTesting_percent{0.00} %"),

                               ("Population", "@Population"),("Population Density", "@Pop_Density")])

p4.add_tools(hover1) # Customization of HoverTool



 #Stylize Legends

legend = (Legend(items=[("Germany", [circle1]),

                        ("Korea, South", [circle2]),

                        ("Finland", [circle3]),

                        ("Denmark", [circle4]),

                        ("Netherlands", [circle5]),

                        ("Australia", [circle6]),

                        ("Croatia", [circle7]),

                        ("Japan", [circle8]),

                        ("Ireland", [circle9]),

                        ("Belarus", [circle10])

                    ]))

legend.background_fill_color = "#1f77b4"

legend.background_fill_alpha = 0

legend.border_line_color = None

legend.click_policy="hide"  # To disable/hide the legend on click

p4.add_layout(legend, "right")



# # Adding Annotations

mytext_Aus1 = Label(x=1075, y=0.37, text='Australia (0.43%)', text_color = "black", text_font_size='8pt', text_font_style = "bold")

# mytext_Aus2 = Label(x=28000, y=0.25, text='(Rate: 0.45%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



p4.add_layout(mytext_Aus1)

# p4.add_layout(mytext_Aus2)



# Adding footers as <div> tags (a division or a section in an HTML document)



header = Div(text = "<b>Positive Tests Rate Vs. Cumulative Confirmed Count (per million) for the Top 10 Countries with the highest Health Capaciy Scores</b>", 

            width=600, style={'font-size': '120%', 'color': 'black'}, align = "start") # height = 50,

subheader = Div(text = "<i>Positive Tests Rate (%) = (Total Confirmed/ Total Tested)*100</i>", 

            width=600, style={'font-size': '90%', 'color': 'black'}, align = "start") # height = 50,



footer1 = Div(text = "<b>Note! <i>Positive Tests Rate is calculated based on the last updated <q>Cumulative Testing Count</q> value of the country by the source and the corresponding number of <q>Cumulative Confirmed Counts</q> that day.</i></b>", 

            width=600, style={'font-size': '80%', 'color': 'black'}, align = "start") # height = 50,



layout=column(header,subheader, p4, footer1)



# Displaying the layout

show(layout) 
# data1.loc[data1["Country"]=="Vietnam"][["Country", "Days", "CumConfirmed", "CumTesting","smoothed_cumulative_testing_count","SmoothedCumTesting_per_CumConfirmed"]]
# Creating empty dataframe to include only latest data

df_tails = pd.DataFrame(columns=["Country", "Days", 

              "CumConfirmed", "CumConfirmed_per_million", 

              "CumDeaths",

              "CumRecovered", 

              "CumTesting","smoothed_cumulative_testing_count",

              "Population", "Pop_Density", "Land_Area_Kmsq", "HealthCapacityScore", "RecoveryRate", "FatalityRate","Active", 

              "SmoothedCumTesting_per_thousand","SmoothedCumTesting_per_CumConfirmed",

              "CumConfirmed_per_SmoothedCumTesting_percent"])

# Feeding values to empty dataframe

for country in data["Country"].unique():

#     df_tails = pd.concat([df_tails, (data1.loc[data["Country"]==country].sort_values(by=['CumTesting', 'CumConfirmed'], ascending=[False, False]).head(1))], ignore_index = True)

    df_tails = pd.concat([df_tails, (data.loc[data["Country"]==country].sort_values(by=['smoothed_cumulative_testing_count', 'CumConfirmed'], ascending=[False, False]).iloc[[1]])], ignore_index = True)  
# MergedData.loc[MergedData["Country/Region"]=="Fiji"][["Country/Region", "Days", "Cumulative Confirmed Count","Cumulative Testing Count",

#                                                      "smoothed_cumulative_testing_count", "SmoothedCumTesting_per_CumConfirmed"]]
# MergedData.loc[MergedData["Country/Region"]=="Taiwan"]
# df_tails["CumTesting"] = df_tails["CumTesting"].fillna(0)

df_tails["smoothed_cumulative_testing_count"] = df_tails["smoothed_cumulative_testing_count"].fillna(0)

df_tails["SmoothedCumTesting_per_thousand"] = df_tails["SmoothedCumTesting_per_thousand"].fillna(0)

df_tails["SmoothedCumTesting_per_CumConfirmed"] = df_tails["SmoothedCumTesting_per_CumConfirmed"].fillna(0)

# df_tails["CumConfirmed_per_SmoothedCumTesting"] = df_tails["CumConfirmed_per_SmoothedCumTesting"].fillna(0)

# df_tails
# print(df_tails['SmoothedCumTesting_per_CumConfirmed'].min(), df_tails['SmoothedCumTesting_per_CumConfirmed'].max())

print(df_tails['CumConfirmed_per_SmoothedCumTesting_percent'].min(), '-',df_tails['CumConfirmed_per_SmoothedCumTesting_percent'].max())
df_tails['CumConfirmed_per_SmoothedCumTesting_percent'].isna().sum()

# df_tails['SmoothedCumTesting_per_CumConfirmed'].isna().sum()
# df_tails[df_tails['CumConfirmed_per_SmoothedCumTesting_percent']<10]
df_tails.isna().sum()
from bokeh.transform import factor_cmap, factor_mark

from bokeh.models import LinearColorMapper, BasicTicker, ColorBar



# Replacing null values in "CumConfirmed_per_SmoothedCumTesting_percent"(positive tests rate) with blanks.

# Also replacing null values in Health Capacity Score column with blanks.

df_tails["CumConfirmed_per_SmoothedCumTesting_percent"] = df_tails["CumConfirmed_per_SmoothedCumTesting_percent"].fillna("")

df_tails["HealthCapacityScore"] = df_tails["HealthCapacityScore"].fillna("")





# 1. Creating column containing Intervals for Population: 



def func(x):

    if x < 1000000:

        return '< 1m'

    if 1000000 < x <= 10000000:

        return '1-10m'

    if 10000000 < x <= 50000000 :

        return '10-50m'

    if 50000000 < x <= 100000000:

        return '50-100m'

    if x > 100000000:

        return '>100m'



df_tails['PopulationInterval'] = df_tails['Population'].apply(func)

# Creating colormap for different levels of Population Intervals

colormap1 = {'< 1m':"silver",'1-10m':"burlywood", '10-50m':"yellow",'50-100m':"orange",'>100m':"red"} # Giving names to colors

# Creating color column for PopulationInterval using list comprehension

df_tails["PopulationColor"] = [colormap1[x] for x in df_tails["PopulationInterval"]]



# 2. Creating column containing Intervals for Health Capacity Score (Score/100): 

def func(x):

    if x == "":

        return 'Data Unavailable'

    if x < 20:

        return '< 20'

    if 20 <= x <= 40:

        return '20-40'

    if 40 <= x <= 60 :

        return '40-60'

    if 60 <= x <= 80:

        return '60-80'

    if 80 <= x <= 100:

        return '80-100'

    

df_tails['ScoreInterval'] = df_tails['HealthCapacityScore'].apply(func)  

# Creating colormap for different levels of Health Capacity Score

colormap2 = {'Data Unavailable':"silver",'< 20':"burlywood",'20-40':"yellow", '40-60':"orange",'60-80':"green",'80-100':"blue"} # Giving names to colors

# Creating color column for ScoreInterval using list comprehension

df_tails["ScoreColor"] = [colormap2[x] for x in df_tails["ScoreInterval"]]



# 3. Creating column containing Intervals for Testing Ratio: 

def func(x):

    if x == 0:

        return 'Zero Testing'

    if 0 < x <= 50:

        return '(0 - 50]'

    if 50 < x <= 100 :

        return '(50 - 100]'

    if 100 < x <= 200:

        return '(100 - 200]'

    if 200 < x <= 400:

        return '(200 - 400]'

    if x > 400:

        return '(400 and more)'

    

df_tails['TestingRateInterval'] = df_tails['SmoothedCumTesting_per_CumConfirmed'].apply(func)

colormap3 = {'Zero Testing':"silver",'(0 - 50]':"burlywood", '(50 - 100]':"yellow",'(100 - 200]':"orange",'(200 - 400]':"red", '(400 and more)': "green" } # Giving names to colors

# Creating color column using list comprehension

df_tails["TestingColor"] = [colormap3[x] for x in df_tails["TestingRateInterval"]]



# 4. Creating column containing Intervals for Positive Testing: 

def func(x):

    if x == "":

        return 'Zero Testing'

    if 0 < x <= 1:

        return '< 1%'

    if 1 < x <= 10:

        return '(1% - 10%]'

    if 10 < x <= 25 :

        return '(10% - 25%]'

    if 25 < x <= 50:

        return '(25% - 50%]'

    if 50 < x <= 75:

        return '(50% - 75%]'

    if 75 < x <= 100:

        return '(75% - 100%]'

    if x > 100:

        return '> 100%'

    

df_tails['TestingPositiveRate'] = df_tails['CumConfirmed_per_SmoothedCumTesting_percent'].apply(func)

colormap4 = {'Zero Testing':"silver",'< 1%':"indigo", '(1% - 10%]':"green",'(10% - 25%]':"yellowgreen",'(25% - 50%]':"yellow", '(50% - 75%]': "orange",'(75% - 100%]':"red",'> 100%':"darkred"  } # Giving names to colors

# Creating color column using list comprehension

df_tails["TestingPositiveRateColor"] = [colormap4[x] for x in df_tails["TestingPositiveRate"]]





# Rearranging the columns in df_tails

df_tails = df_tails[["Country", "Days", 

                     "CumConfirmed", "CumConfirmed_per_million",

                     "CumDeaths", 

                     "CumRecovered", 

                     "CumTesting", "smoothed_cumulative_testing_count",

                     "Population", "Pop_Density", "Land_Area_Kmsq", "HealthCapacityScore",

                     "Active", "RecoveryRate","FatalityRate",

                     "SmoothedCumTesting_per_CumConfirmed", "SmoothedCumTesting_per_thousand",

                     'CumConfirmed_per_SmoothedCumTesting_percent',

                     'TestingPositiveRate', 'TestingPositiveRateColor',

                     "PopulationInterval", "PopulationColor", "ScoreInterval", "ScoreColor", "TestingRateInterval", "TestingColor"

                    ]]
# df_test["Country/Region"].unique()
# df_tails
# data1.loc[data1["Country"]=="Vietnam"]

# df_tails.loc[df_tails["Country"]=="Vietnam"]
plots = [] 

x_list = ["Population", "Pop_Density"]



for x in x_list:

    # Creating the figure

    p4 = figure(x_axis_label = x , y_axis_label = 'Confirmed Cases', y_axis_type ="log",x_axis_type ="log",

                title =('Cumulative Confirmed Counts Vs. ' + x + ' for the Countries')) #creating figure object 

#     p4.add_layout(Title(text="Confirmed Cases Vs. ' + x + ' for the Countries', text_font_size="11pt", above, 'center'))

    p4.add_layout(Title(text="Rate of Positive Tests (PT%) = (Total Confirmed/ Total Tested)*100", text_font_size="9pt", align="left"), 'above')

    p4.add_layout(Title(text="Grouping of countries is by Intervals of Positive Tests Rates", text_font_size="10pt", text_font_style="italic", align="left"), 'above')



    # Plotting the graph

    c0 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=="Zero Testing"]))

    c1 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='< 1%']))

    c2 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='(1% - 10%]']))

    c3 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='(10% - 25%]']))

    c4 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='(25% - 50%]']))

    c5 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='(50% - 75%]']))

    c6 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='(75% - 100%]']))

    c7 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.65, color ="TestingPositiveRateColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingPositiveRate"]=='> 100%']))

    

    # Stylize the plot area

    p4.plot_width = 650                     # To change the width of the plot

    p4.plot_height = 400                    # To change the height of plot

    p4.background_fill_color = "ghostwhite"   # To add background colorto the figure

    # p.background_fill_alpha = 0.12

    

    # Stylize the title

#     p4.title.text = "Title With Options"

    p4.title.align = "left"

    p4.title.text_color = "red"

    p4.title.text_font_size = "10px"

#     p4.title.background_fill_color = "#aaaaee"





    # Stylize the grid

    p4.xgrid.grid_line_color = "white"

    p4.ygrid.grid_line_color = "white"

    p4.ygrid.grid_line_alpha = 0.7

    p4.xgrid.grid_line_dash = [5,3]

    p4.xgrid.grid_line_width = 2

    p4.grid.grid_line_dash = [5,3]

    p4.grid.grid_line_width = 2





    # Axes Geometry

    # p4.x_range = DataRange1d(start = 0, end = 1000 ) # x_range changes according to the active legends

    p4.y_range = DataRange1d(start = 1, only_visible=True) # y_range changes according to the active legends

    p4.x_range = DataRange1d(start = 1, only_visible=True)

#     p4.yaxis.formatter.use_scientific = False

#     p4.xaxis.formatter.use_scientific = False



    # Stylize the axes

#     p4.xaxis.major_label_orientation = "vertical"

    p4.yaxis.ticker.desired_num_ticks = 8 

    p4.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p4.axis.axis_label_text_color = "black"

    # p4.xaxis.major_label_orientation = "vertical"

    # p.axis.major_label_text_font_style = "bold" # for axis' major tick marks' labels value color

    p4.yaxis.minor_tick_in = 0

    p4.yaxis.minor_tick_out = 0

    p4.xaxis.minor_tick_in = 0

    p4.xaxis.minor_tick_out = 0





    # Stylize the figure title

    p4.title.text_color = "black"

    p4.title.text_font = "times"

    p4.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p4.title.align = "center"



    legend = Legend(items=[('< 1%', [c1]),("(1% - 10%]", [c2]), 

                           ("(10% - 25%]", [c3]), ("(25% - 50%]" , [c4]),

                           ("(50% - 75%]" , [c5]), ("(75% - 100%]" , [c6]),

                           ("> 100%" , [c7]), ('Zero Testing', [c0])

                  ])

#     legend.title = "Select Testing Ratio Interval:"

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p4.add_layout(legend, 'right')



    # Adding arrow to locate marker for Australia in the large number of points

    from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead

    # Arrow for Australia

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray", line_dash = [1,3], size = 6),

                   x_start=24000000, y_start=27000, x_end=24000000, y_end=750000, line_dash = [3,1]))

    # Arrow for New Zealand

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray", line_dash = [1,3], size = 6),

                   x_start=4800000, y_start=1500, x_end=4800000, y_end=200000, line_dash = [3,1]))

#     # Arrow for Korea, South

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray",line_dash = [1,3], size = 6),

                   x_start=53000000, y_start=18562, x_end=200000000, y_end=18562, line_dash = [3,1]))

#     # Arrow for Taiwan

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray",line_dash = [1,3], size = 6),

                   x_start=23000000, y_start=470, x_end=23000000, y_end=10, line_dash = [3,1]))

    # Arrow for Vietnam

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray", line_dash = [1,3], size = 6),

                   x_start=100000000, y_start=220, x_end=100000000, y_end=50, line_dash = [3,1]))

    # Arrow for US

    p4.add_layout(Arrow(end=VeeHead(line_color="gray", line_dash = [1,3], size = 6),

                   x_start=300000000, y_start=5200000, x_end=50000000, y_end=5200000, line_dash = [3,1])) #,line_width=2

#     # Arrow for Brazil

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray", line_dash = [1,3], size = 6),

                   x_start=220000000, y_start=3000000, x_end=335000000, y_end=3000000, line_dash = [3,1])) # , line_width=2

    

     # Adding Annotations

    mytext_Aus1 = Label(x=8700000, y=1180000, text='Australia', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Aus2 = Label(x=8700000, y=730000, text='(PT%: 0.43%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



    mytext_NZ1 = Label(x=950000, y=315000, text='New Zealand', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_NZ2 = Label(x=950000, y=190000, text='(PT%: 0.24%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")

    

    mytext_KSouth1 = Label(x=215700000, y=11562, text='South Korea', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_KSouth2 = Label(x=215000000, y=6562, text='(PT%: 1.05%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



    mytext_Taiwan1 = Label(x=10000000, y=5, text='Taiwan', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Taiwan2 = Label(x=10000000, y=2.5, text='(PT%:0.56%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



    mytext_Vietnam1 = Label(x=41000000, y=25, text='Vietnam', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Vietnam2 = Label(x=40900000, y=14, text='(PT%: 0.11%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



    mytext_US1 = Label(x=22000000, y=4000000, text='US', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_US2 = Label(x=16000000, y=2400000, text='(PT%: 7.47%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



    mytext_Brazil1 = Label(x=345000000, y=1800000, text='Brazil', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Brazil2 = Label(x=235000000, y=1000000, text='(PT%: 75.06%)', text_color = "black", text_font_size='7pt', text_font_style = "italic")



    p4.add_layout(mytext_Aus1)

    p4.add_layout(mytext_Aus2)

    p4.add_layout(mytext_NZ1)

    p4.add_layout(mytext_NZ2)

    p4.add_layout(mytext_KSouth1)

    p4.add_layout(mytext_KSouth2)

    p4.add_layout(mytext_Taiwan1)

    p4.add_layout(mytext_Taiwan2)

    p4.add_layout(mytext_Vietnam1)

    p4.add_layout(mytext_Vietnam2)

    p4.add_layout(mytext_US1)

    p4.add_layout(mytext_US2)

    p4.add_layout(mytext_Brazil1)

    p4.add_layout(mytext_Brazil2)

#     # Stylize the tools

    # Adding customization to HoverTool

    p4.tools = [ZoomInTool(),ZoomOutTool(), ResetTool(), BoxZoomTool()]

    hover1 = HoverTool(tooltips = [("Country", "@Country"),("Total Confirmed", "@CumConfirmed"),

                                  ("Total Tested", "@smoothed_cumulative_testing_count"), 

                                  ("Positive Testing Ratio", "@CumConfirmed_per_SmoothedCumTesting_percent{0.00} %"),

                                  ("Population", "@Population"),("Population Density", "@Pop_Density")])

    p4.add_tools(hover1) # Customization of HoverTool

    plots.append(p4)

    



# Creating separate tabs for the two figures created above

tab1 = Panel(child=plots[0], title="Population")

tab2 = Panel(child=plots[1], title="Population Density")



# Creating list of Tabs

tabs = Tabs(tabs=[ tab1, tab2 ])  



# Adding footers as <div> tags (a division or a section in an HTML document)



footer1 = Div(text = "<b>Note! Positive Tests Rate is calculated based on the last updated <q>Cumulative Testing Count</q> value of the country and the corresponding number of <q>Cumulative Confirmed Counts</q> that day.</b>", 

            width=600, style={'font-size': '65%', 'color': 'black'}, align = "start") # height = 50,

footer2 = Div(text = "<b>For example, for Vietnam the <q>Cumulative Testing Count</q> was last updated on day 84 <i> (since first 10 confirmed cases)</i> but currently (at the time of plotting the graph) country is at day 186 but no testing data has been updated after day 84.</b>", 

            width=600, style={'font-size': '65%', 'color': 'black'}, align = "start") # height = 50,

footer3 = Div(text = "<b>As mentioned in the testing dataset source website <i>(https://ourworldindata.org/coronavirus-testing)</i>: Until 29th April (day 84-(since first 10 confirmed cases)), the Vietnamese Ministry of Health were updating a figure for tests on its disease situation statistics page daily. More recently the website stopped updating its testing data, and finally stopped reporting the figures altogether. The last date we were able to collect testing data was on the 29th April.<b>", 

            width=600, style={'font-size': '65%', 'color': 'black'}, align = "start") # height = 50,



layout=column(tabs, footer1, footer2, footer3)



# Displaying the layout

show(layout) 
plots = [] 

x_list = ["Population", "Pop_Density"]



for x in x_list:

    # Creating the figure

    p4 = figure(x_axis_label = x , y_axis_label = 'Confirmed Cases', y_axis_type ="log",x_axis_type ="log",

                title =('Cumulative Confirmed Counts Vs. ' + x + ' for the Countries')) #creating figure object 

#     p4.add_layout(Title(text="Confirmed Cases Vs. ' + x + ' for the Countries', text_font_size="11pt", above, 'center'))

    p4.add_layout(Title(text="Testing Ratio (TR) = Total Tested / Total Confirmed", text_font_size="9pt", align="left"), 'above')

    p4.add_layout(Title(text="Grouping of countries is by Testing Ratio Interval", text_font_size="10pt", text_font_style="italic", align="left"), 'above')



    # Plotting the graph

    c0 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.7, color ="TestingColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingRateInterval"]=="Zero Testing"]))

    c1 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.7, color ="TestingColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingRateInterval"]=='(0 - 50]']))

    c2 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.7, color ="TestingColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingRateInterval"]=='(50 - 100]']))

    c3 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.7, color ="TestingColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingRateInterval"]=='(100 - 200]']))

    c4 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.7, color ="TestingColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingRateInterval"]=='(200 - 400]']))

    c5 = p4.circle(x=x, y = "CumConfirmed", size =10, fill_alpha = 0.7, color ="TestingColor",line_dash = [3,3], 

                   source = ColumnDataSource(df_tails[df_tails["TestingRateInterval"]=='(400 and more)']))

    

    # Stylize the plot area

    p4.plot_width = 650                     # To change the width of the plot

    p4.plot_height = 400                    # To change the height of plot

    p4.background_fill_color = "ghostwhite"   # To add background colorto the figure

    # p.background_fill_alpha = 0.12

    

    # Stylize the title

#     p4.title.text = "Title With Options"

    p4.title.align = "left"

    p4.title.text_color = "red"

    p4.title.text_font_size = "10px"

#     p4.title.background_fill_color = "#aaaaee"





    # Stylize the grid

    p4.xgrid.grid_line_color = "white"

    p4.ygrid.grid_line_color = "white"

    p4.ygrid.grid_line_alpha = 0.7

    p4.xgrid.grid_line_dash = [5,3]

    p4.xgrid.grid_line_width = 2

    p4.grid.grid_line_dash = [5,3]

    p4.grid.grid_line_width = 2





    # Axes Geometry

    # p4.x_range = DataRange1d(start = 0, end = 1000 ) # x_range changes according to the active legends

    p4.y_range = DataRange1d(start = 1, only_visible=True) # y_range changes according to the active legends

    p4.x_range = DataRange1d(start = 1, only_visible=True)

#     p4.yaxis.formatter.use_scientific = False

#     p4.xaxis.formatter.use_scientific = False



    # Stylize the axes

#     p4.xaxis.major_label_orientation = "vertical"

    p4.yaxis.ticker.desired_num_ticks = 8 

    p4.axis.axis_label_text_font_style = "bold" #"normal", "italic"

    p4.axis.axis_label_text_color = "black"

    # p4.xaxis.major_label_orientation = "vertical"

    # p.axis.major_label_text_font_style = "bold" # for axis' major tick marks' labels value color

    p4.yaxis.minor_tick_in = 0

    p4.yaxis.minor_tick_out = 0

    p4.xaxis.minor_tick_in = 0

    p4.xaxis.minor_tick_out = 0





    # Stylize the figure title

    p4.title.text_color = "black"

    p4.title.text_font = "times"

    p4.title.text_font_size = "20px" # px stands for pixel. Have to mention.

    p4.title.align = "center"



    legend = Legend(items=[('Zero Testing', [c0]),('(0 - 50]', [c1]),("(50 - 100]", [c2]), 

                           ("(100 - 200]", [c3]), ("(200 - 400]" , [c4]), 

                           ("(400 and more)" , [c5])

                  ])

#     legend.title = "Select Testing Ratio Interval:"

    legend.border_line_color = None

    legend.click_policy="hide"  # To disable/hide the legend on click

    p4.add_layout(legend, 'right')



    # Adding arrow to locate marker for Australia in the large number of points

    from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead

    # Arrow for Australia

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray", line_dash = [1,3], size = 6),

                   x_start=24000000, y_start=27000, x_end=24000000, y_end=750000))

    # Arrow for New Zealand

    p4.add_layout(Arrow(end=VeeHead(line_color="darkgray", line_dash = [1,3], size = 6),

                   x_start=4800000, y_start=1500, x_end=4800000, y_end=200000))

#     # Arrow for Lithania

    p4.add_layout(Arrow(end=VeeHead(line_color="black", line_width=2, size = 6),

                   x_start=2700000, y_start=2080, x_end=2700000, y_end=10))

#     # Arrow for Fiji

    p4.add_layout(Arrow(end=VeeHead(line_color="black", line_width=2, size = 6),

                   x_start=800000, y_start=25, x_end=200000, y_end=5))

    # Arrow for Vietnam

    p4.add_layout(Arrow(end=VeeHead(line_color="black", line_width=2, size = 6),

                   x_start=100000000, y_start=220, x_end=100000000, y_end=50))

    

     # Adding Annotations

    # Adding Annotations

    mytext_Aus1 = Label(x=8700000, y=1180000, text='Australia', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Aus2 = Label(x=8700000, y=730000, text='(TR: 232.46)', text_color = "black", text_font_size='7pt', text_font_style = "bold")



    mytext_NZ1 = Label(x=950000, y=315000, text='New Zealand', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_NZ2 = Label(x=950000, y=190000, text='(TR: 422.63)', text_color = "black", text_font_size='7pt', text_font_style = "bold")

    

    mytext_Lithuania = Label(x=1000000, y=5, text='Lithuania (TR: 225.87)', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Fiji = Label(x=90000, y=2.5, text='Fiji (TR: 284.32)', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    mytext_Vietnam = Label(x=41000000, y=25, text='Vietnam (TR: 879.16)', text_color = "black", text_font_size='8pt', text_font_style = "bold")

    

    p4.add_layout(mytext_Aus1)

    p4.add_layout(mytext_Aus2)

    p4.add_layout(mytext_NZ1)

    p4.add_layout(mytext_NZ2)

    p4.add_layout(mytext_Lithuania)

    p4.add_layout(mytext_Fiji)

    p4.add_layout(mytext_Vietnam)

    # Stylize the tools

    # Adding customization to HoverTool

    p4.tools = [ZoomInTool(),ZoomOutTool(), ResetTool(), BoxZoomTool()]

    hover = HoverTool(tooltips = [("Country", "@Country"),("Total Confirmed", "@CumConfirmed"),

                                  ("Total Tested", "@smoothed_cumulative_testing_count"), 

                                  ("Testing Ratio", "@SmoothedCumTesting_per_CumConfirmed{0.000}"),

                                  ("Population", "@Population"),("Population Density", "@Pop_Density")])

    p4.add_tools(hover) # Customization of HoverTool

    plots.append(p4)

    



# Creating separate tabs for the two figures created above

tab1 = Panel(child=plots[0], title="Population")

tab2 = Panel(child=plots[1], title="Population Density")



# Creating list of Tabs

tabs = Tabs(tabs=[ tab1, tab2 ])  



# Adding footers as <div> tags (a division or a section in an HTML document)



footer1 = Div(text = "<b>Note! Testing Ratio is calculated based on the last updated <q>Cumulative Testing Count</q>by the country and the corresponding number of <q>Cumulative Confirmed Counts</q> that day.</b>", 

            width=600, style={'font-size': '65%', 'color': 'black'}, align = "start") # height = 50,

footer2 = Div(text = "<b>For example, for Vietnam the <q>Cumulative Testing Count</q> was last updated on day 84 <i> (since first 10 confirmed cases)</i> but currently (at the time of plotting the graph) country is at day 176 but no testing data has been updated after day 84.</b>", 

            width=600, style={'font-size': '65%', 'color': 'black'}, align = "start") # height = 50,

footer3 = Div(text = "<b>As mentioned in the testing dataset source website <i>(https://ourworldindata.org/coronavirus-testing)</i>: Until 29th April (day 84-(since first 10 confirmed cases)), the Vietnamese Ministry of Health were updating a figure for tests on its disease situation statistics page daily. More recently the website stopped updating its testing data, and finally stopped reporting the figures altogether. The last date we were able to collect testing data was on the 29th April.<b>", 

            width=600, style={'font-size': '65%', 'color': 'black'}, align = "start") # height = 50,



layout=column(tabs, footer1, footer2, footer3)



# Displaying the layout

show(layout) 