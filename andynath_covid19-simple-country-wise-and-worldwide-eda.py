import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sys

np.set_printoptions(threshold=sys.maxsize)
path_train = 'covid19-global-forecasting-week-2/train.csv'

path_test = 'covid19-global-forecasting-week-2/test.csv'

path_sbumit = 'covid19-global-forecasting-week-2/submission.csv'



train_kaggle = '/kaggle/input/covid19-global-forecasting-week-2/train.csv'

test_kaggle = '/kaggle/input/covid19-global-forecasting-week-2/test.csv'

submit_kaggle = '/kaggle/input/covid19-global-forecasting-week-2/submission.csv'



df_train = pd.read_csv(train_kaggle)

df_test = pd.read_csv(test_kaggle)

submission = pd.read_csv(submit_kaggle)
df_train.rename(columns = {'Country_Region': 'Country/Region', 'Province_State':'Province/State'}, inplace = True)
df_test.rename(columns = {'Country_Region': 'Country/Region', 'Province_State':'Province/State'}, inplace = True)
# Dataset Dimesnions

print('Train shape', df_train.shape)

print('Test shape', df_test.shape)

# Missing/Null Values

print('\nTrain Missing\n', df_train.isnull().sum())

print('\nTest Missing\n', df_test.isnull().sum())
lst = df_train['Country/Region'].unique()

#print('Total_Countries\n:', len(lst))

#for i in lst:

#    print(i)
print(df_train['Date'].min(), ' - ', df_train['Date'].max())
# GroupBy syntax (columns to group by in list)[Columns to aggregate, apply function to] . aggregation functions on it 

train_cases_conf = df_train.groupby(['Date'])['ConfirmedCases'].sum()

#train_cases_conf
train_cases_conf.plot(figsize = (10,8), title = 'Worldwide Confirmed Cases', grid = True)
train_cases_conf.plot(figsize = (10,8), title = 'Worldwide Confirmed Cases (Log)', grid = True, logy = True)
train_fatal = df_train.groupby(['Date'])['Fatalities'].sum()

#train_fatal
train_fatal.plot(figsize = (10,8), title = 'Worldwide Fatalaties', grid = True)
# To DO :

# 1. Add option to check for World or Country (Done)

# 2. Add toggle for scale = "linear", "log"

# 3. Check if country is present in Dataset, else throw error.



def country_stats(country, df):

    if country != 'World':

        country_filt = (df['Country/Region'] == country)

        df_cases = df.loc[country_filt].groupby(['Date'])['ConfirmedCases'].sum()

        df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()

    else:

        df_cases = df.groupby(['Date'])['ConfirmedCases'].sum()

        df_fatal = df.groupby(['Date'])['Fatalities'].sum()

        

    fig, axes = plt.subplots(nrows = 2, ncols= 1, figsize=(15,15))

    axes[0].set_title(country + ' Confirmed Cases')

    axes[1].set_title(country + ' Fatalities')

    df_cases.plot(ax = axes[0])

    df_fatal.plot(ax = axes[1])

    

country_stats('India', df_train)
# grouping using same Country filter to get fatalities on each date (grouped by date)

# groupby([list of columns to groupby]) [which columns to apply aggregate fx to ]. (aggregate function)

# To Do - Fix Ticks 



def country_stats_log(country, df):

    count_filt =(df_train['Country/Region'] == country)

    df_count_case = df_train.loc[count_filt].groupby(['Date'])['ConfirmedCases'].sum()

    df_count_fatal = df_train.loc[count_filt].groupby(['Date'])['Fatalities'].sum()

    plt.figure(figsize=(15,10))

    plt.axes(yscale = 'log')

    plt.plot(df_count_case.index, df_count_case.tolist(), 'b', label = country +' Total Confirmed Cases')

    plt.plot(df_count_fatal.index, df_count_fatal.tolist(), 'r', label = country +' Total Fatalities')

    plt.title(country +' COVID Cases and Fatalities (Log Scale)')

    plt.legend()

    



country_stats_log('India', df_train)
# as_index = False to not make the grouping column the index, creates a df here instead of series, preserves

# Confirmedcases column



# Confirmed Cases till a particular day by country



def case_day_country (Date, df):

    df = df.groupby(['Country/Region', 'Date'], as_index = False)['ConfirmedCases'].sum()

    date_filter = (df['Date'] == Date)

    df = df.loc[date_filter]

    df.sort_values('ConfirmedCases', ascending = False, inplace = True)

    sns.catplot(x = 'Country/Region', y = 'ConfirmedCases' , data = df.head(10), height=5,aspect=3, kind = 'bar')

    

    

case_day_country('2020-03-25', df_train)
def case_country_diff (Date, df):

    

    df = df.groupby(['Country/Region', 'Date'], as_index = False)['ConfirmedCases'].sum()

    

    # Creating the two filter for date + dateoffset(1)

    date_filter = (df['Date'] == Date)

    # Getting filter for previous day 

    arr_date = Date.split('-')

    arr_date[2] = str(int(arr_date[2]) - 1) 

    new_date = '-'.join(arr_date)

    date_filter_2 = (df['Date'] == new_date)

    

    # Creating the two dataframes for the dates

    # Have to reset index, groupby screwed them up.

    df_1 = df.loc[date_filter]

    df_1.reset_index(drop= True, inplace = True)

    df_2 = df.loc[date_filter_2]

    df_2.reset_index(drop= True, inplace = True)

    

    # Getting change in Cases 

    df_1['ChangeCases'] = df_1['ConfirmedCases'] - df_2['ConfirmedCases']

    df_1.sort_values('ChangeCases', ascending = False, inplace = True)

    sns.catplot(x = 'Country/Region', y = 'ChangeCases' , data = df_1.head(10), height=5,aspect=3, kind = 'bar')





case_country_diff('2020-03-25', df_train)
# Plotting a simple logistic curve using numpy and matplotlib.

# x = (-6,6), L =1, k = 1, x0 =0







x = np.arange(-6,7)

power = -1*x

y = 1 / (1 + np.exp(power))



plt.figure(figsize=(15,7))

plt.title('Simple Logistic Curve')

plt.grid(True)

plt.plot(x, y)

plt.show()
country_stats('China', df_train)
# pandas df.shift to shift dataframe



# Getting Worldwide Growth Factor

# GF = cases on date/ cases on day -1



growth_factor = train_cases_conf/train_cases_conf.shift(1)

growth_factor.plot(grid = True, title = 'Worlwide Growth Factor', figsize = (18,6))