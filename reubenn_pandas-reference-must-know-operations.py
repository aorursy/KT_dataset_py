import pandas as pd

import numpy as np # importing np since you will have to use it at one point or the other with pandas!
# check pandas versions - NOTE THAT ALL PANDAS OPERATIONS ARE DEPENDANT ON THE VERSION YOU USE.

print(pd.__version__)

print(np.__version__)
# create empty dataframe, specifying column names

df = pd.DataFrame(columns=['X', 'Y', 'Z'])

df
# create dataframe from csv-formatted string

from io import StringIO

dfFromString = pd.read_csv(StringIO("Employee ID,Name,Age,Grade,Marital Status,Number of kids\r\n1,Chris,M,38,B,Married,1\r\n2,Mira,F,33,A,Single,0"))



# pd.head() returns the first few rows of a df, 5 by default ( tail() will return the last few )

print("Head:")

print(dfFromString.head())



# pd.shape returns the number of rows by columns

print("\nShape:")

print(dfFromString.shape)      



# pd.size returns number of rows * number of columns

print("\nSize:")

print(dfFromString.size)



# pd.dtypes returns types of each column

print("\ndtypes:")

print(dfFromString.dtypes)
# create dataframe from list of lists, specifying column names

df = pd.DataFrame.from_records([[100,200,300,], [312,412,512], [689,789,889]], columns=['Col0', 'Col1', 'Col2'])

df.head()
# create dataframe from dictionary, specifying column names

dictData =  {'Harry': 38, 'Mary': 30,'Ben': 12,}

dfDateVal = pd.DataFrame(list(dictData.items()), columns=['Name', 'Age'])

dfDateVal
# read CSV file online/ using weblink

data1 = pd.read_csv("https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv")

data1.head()
# read in data specifying seperator/delimiter

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_2.csv', sep='|')

employeeData.head()
# read in a file not having a header

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_3.csv', header=None)

employeeData.head()
# read in file specifying column types

data1 = pd.read_csv("/kaggle/input/all-pandas-operations-reference/Games.csv", dtype={'Game Number':float})

data1.dtypes
# read in file skipping some lines

apartmentData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Apartments.csv', skiprows=3)

apartmentData
# specify column names while reading in CSV

col_names = ['Country_Name', 'Num_of_people', 'Language_Spoken']

countryDF = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv', names=col_names, header=0)

countryDF.head()
# read only a few of many columns

empData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv', usecols=['ID', 'Name', 'Gender'])

empData.head()
# for using less memory, read in a column of repeating strings as type 'category;

empData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv', dtype={'Marital Status':'category'})

empData.dtypes
# specify missing value labels while reading in data

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

employeeData.tail(10) # print last few rows
# read in data having column of date time strings, parsing the column

employee_joining = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Joining.csv', parse_dates=['Joining Datetime'])

print("dtypes:\n", employee_joining.dtypes)



print("\nDF:\n", employee_joining.head())
# filter DF by a particular string in a string-type column

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

employeeData[employeeData.Name.str.contains('ba')].head()
# basic info about a df

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

employeeData.info()
# statistical info about df

employeeData.describe()
# see types of columns

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

dtype_df = employeeData.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
# print unique values in a column

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

employeeData.Grade.unique()
# print number of unique values in a column

employeeData.Grade.nunique()
# print counts for each unique value in a column, including NA

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

employeeData['Kids'].value_counts(dropna=False)
# percentage of each unique value in a column, including NA

employeeData['Kids'].value_counts(dropna=False, normalize=True) * 100
# crosstable of 2 columns

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

pd.crosstab(employeeData.Gender, employeeData.Grade)
# get a single statistic for a column

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

employeeData.groupby('Grade').Kids.mean()
# get multiple statistics for a column

employeeData.groupby('Grade').Kids.agg(['min', 'max', 'mean'])
# count how many rows have no missing values

employeeData.dropna().shape[0]
# location/index/position of all null values

employeeData.isnull().head(20)
# number of null values by columns

employeeData.isnull().sum()
# print proportion of missing values per column, sorted

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])



percent_missing = employeeData.isnull().sum() * 100 / len(employeeData)

missing_value_df = pd.DataFrame({'column_name': employeeData.columns, 'percent_missing': percent_missing}).sort_values('percent_missing').set_index('column_name')

missing_value_df
# select using loc & ":" 

# loc is inclusive on both sides

employeeData = pd.read_csv("/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv")

employeeData.loc[0:3] # if columns location not specified, implicit mode used - all columns
# select using loc

# 'loc' does not accept integers for columns; column names have to be used.

employeeData.loc[0:3, ['Name','Marital Status']]
# select using iloc

# unlike 'loc', 'iloc' accepts integers for columns

# iloc is exclusive on right side, so 1:4 means 1,2,3

employeeData.iloc[0:3, 2:4]
# select columns from a list of strings

employeeData[['Name','Kids']].head()
# select string-type single column using column name, on a single value

data2 = employeeData[employeeData.Name=='Henry']

data2.head()
# select string-type single column on multiple values

data2 = employeeData.copy()

data2 = data2[data2["Name"].isin(["Henry", "Ava"])]

data2.head()
# select rows NOT containing a particular string

data2 = employeeData.copy()

data2 = data2[~data2["Name"].str.contains("o")] # exclude names containing 'o'

data2.head()
# select on multiple columns

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

employeeData[(employeeData["Kids"] == 3) & (employeeData["Marital Status"] == 'Single')].head()
# select rows having missing values in a column

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

employeeData[employeeData.Kids.isnull()].head()
# use a column as index, and then as a row-label for filtering

# this column needs to have non-missing unique values

countryInfo1 = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print('\n------- Before setting index ---------- \n', countryInfo1.head())



countryInfo1.set_index('Country', inplace=True)

print('\n------- After setting index ---------- \n', countryInfo1.head())



print("\nCanada's population: {}".format(countryInfo1.loc['Canada', 'Population']))
# sort by index

countryInfo1.sort_index()
# selecting rows on timestamps

employee_joining = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Joining.csv', parse_dates=['Joining Datetime'])

employee_joining.loc[employee_joining['Joining Datetime'] >= pd.to_datetime('1/1/1933'), :]
# add a new column having constant values

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print("countryInfo before:\n", countryInfo.head())



countryInfo["Population density"] = 20000

print("\ncountryInfo after:\n", countryInfo.head())
# create a new column based on conditions of existing column(s)

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

print("employeeData before:\n", employeeData.head())



employeeData['isMale'] = np.where(employeeData['Gender']=='M', 'yes', 'no')

print("\nemployeeData after:\n", employeeData.head())
# rename columns - using dictionary to specify old & new names

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print("countryInfo before:\n", countryInfo.head())



countryInfo.rename(columns={'Population':'Pop.'}, inplace=True)



print("\ncountryInfo after:\n", countryInfo.head())
# rename columns - using a list to rename all columns

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print("countryInfo before:\n", countryInfo.head())



countryInfo.columns = ['Cntry', 'Pop.', 'LANG.']



print("\ncountryInfo after:\n", countryInfo.head())
# rename multiple column names using single operation

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

countryInfo.columns = ['Country_Name', 'Country_Population', 'Country_Language']

print("countryInfo before:\n", countryInfo.head())



# replace th string "Country_" with "C_"

countryInfo.columns = countryInfo.columns.str.replace('Country_', 'C_')



print("\ncountryInfo after:\n", countryInfo.head())
# remove a single column

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print("countryInfo before:\n", countryInfo.head())



countryInfo.drop('Population', axis=1, inplace=True)



print("\ncountryInfo after:\n", countryInfo.head())
# remove multiple columns -  using a list of names of columns

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

print("employeeData before:\n", employeeData.head())



employeeData.drop(['Hobbies', 'Kids'], axis=1, inplace=True)



print("\nemployeeData after:\n", employeeData.head())
# remove multiple columns - using positions of columns

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

print("employeeData before:\n", employeeData.head())



employeeData.drop(data2.columns[3:], axis=1, inplace=True)



print("\nemployeeData after:\n", employeeData.head())
# change a column type

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print("countryInfo.dtypes before:\n", countryInfo.dtypes)



countryInfo.Population = countryInfo.Population.astype(float)



print("\ncountryInfo.dtypes after:\n", countryInfo.dtypes)
# convert all object-type columns to categorical

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

print("countryInfo.dtypes before:\n", countryInfo.dtypes)



lisCatCols = ['Country', 'Main Language']

countryInfo[lisCatCols] = countryInfo[lisCatCols].astype('category')



print("\ncountryInfo.dtypes after:\n", countryInfo.dtypes)
# change a column type, making misformatted data NaN

countryInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language_with_errors.csv')

print("countryInfo before:\n", countryInfo.head())

print("\ncountryInfo.dtypes before:\n", countryInfo.dtypes)



countryInfo.Population = pd.to_numeric(countryInfo.Population, errors='coerce')



print("\n\ncountryInfo after:\n", countryInfo.head())

print("\ncountryInfo.dtypes after:\n", countryInfo.dtypes)
# add a new column for rows having similar column value(s)

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

print("employeeData before:\n", employeeData.head())



# for each row, add a column of total people having the same number of kids

employeeData['other'] = employeeData.groupby('Kids')['Kids'].transform('count')



print("\nemployeeData after:\n", employeeData.head())
# drop rows having missing data in any col

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

print("Df Shape before: ", employeeData.shape)

employeeData.dropna(how='any', inplace=True) # use 'all' for checking all columns

print("Df Shape after: ", employeeData.shape)
# drop NAs in any of specified columns

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

print("Df Shape before: ", employeeData.shape)

employeeData.dropna(subset=['Kids', 'Hobbies'], how='any', inplace=True)

print("Df Shape after: ", employeeData.shape)
# replace NAs in a column with a value

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

employeeData['Hobbies'].fillna(value='Not reported', inplace=True)



# check that substitution took place

employeeData[employeeData['Hobbies'] == 'Not reported'].head()
# replace NAs in all columns of dataframe by median of corresponding column

## df = pd.read_csv(.........)

## df = df.fillna(df.median())
# drop columns having percentage of missing values greater than 80

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_4_with_NAs.csv', na_values=['', "NA", "--"])

print("Df Shape before: ", employeeData.shape)



# drop columns having more than 5% data missing

employeeData = employeeData.loc[:, employeeData.isnull().mean() < .05]



print("Df Shape after: ", employeeData.shape)
# capitalise all entries of string column

employeeData.Name = employeeData.Name.str.upper()

employeeData.Name.head()
# replace a particular string

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

employeeData.Name = employeeData.Name.str.replace('ba', 'za') # this will affect the name 'Sebastian'



# check for names containing 'za' now

employeeData[employeeData.Name.str.contains('za')]
# get mean of columns using 'axis' parameter, this will work on ONLY numeric columns/rows

employeeData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

employeeData.mean() # axis = 0, if unspecified
# convert column of timestamp strings to datetime

employee_joining = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Joining.csv')

employee_joining['Joining Datetime'] = pd.to_datetime(employee_joining['Joining Datetime'])

print("dtypes:\n", employee_joining.dtypes)
# convert a column of timestamp strings to datetime, specifying format

employee_joining = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Joining.csv')

employee_joining['Joining Datetime'] = pd.to_datetime(employee_joining['Joining Datetime'],

                                                      format="%m/%d/%Y %H:%M")

employee_joining.dtypes
# to see time difference between consecutive rows of a datetime column

employee_joining['Joining Datetime'].diff()
# extract year, month, day, day of year, week, hour, minute

employee_joining['Joining Datetime'].dt.hour
# difference between timestamps, extract days

(employee_joining['Joining Datetime'].max() - employee_joining['Joining Datetime'].min()).days
# add a new column for seconds since 1st Jan 1900 (For Epoch i.e. 1970, change appropriately)

employee_joining = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Joining.csv', parse_dates=['Joining Datetime'])



employee_joining['EpochTime'] = (employee_joining['Joining Datetime'] - pd.Timestamp("1900-01-01")) // pd.Timedelta('1s')

employee_joining.head()
# set datetime column as index, filter rows between string timestamps

employee_joining = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Joining.csv', parse_dates=['Joining Datetime'])



temp_emp = employee_joining.copy()

temp_emp.set_index('Joining Datetime', inplace=True)



temp_emp.loc['1930-06-01 22:00:00':'1931-12-31 19:00:00']
# find any duplicated rows

orgData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/OrgData.csv')

orgData.duplicated() # returns a Series of booleans
# get number of duplicates

orgData.duplicated().sum()
# to see rows which are duplicated

orgData.loc[orgData.duplicated(keep='first'), :] # show first occurence of dup
# drop duplicate rows

print('size of OrgData before dropping: {}'.format(orgData.shape))



orgData.drop_duplicates(inplace=True)



print('size of OrgData after dropping: {}'.format(orgData.shape))
# use only certain columns when checking for duplication

orgData.duplicated(subset=['Name', 'Gender'])
# check whether duplicate columns exist, keep only the first, delete the rest

companyInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/CompanyInfo.csv')

print('size of companyInfo before dropping dup. columns: {}'.format(companyInfo.shape))



companyInfo = companyInfo.loc[:,~companyInfo.T.duplicated(keep='first')]

print('size of companyInfo after dropping dup. columns: {}'.format(companyInfo.shape))
# drop columns with constant values( only 1 value) throughout



orgData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/OrgData.csv')



# make a column having constant/same value

orgData["const_col"] = 23



print("Dataframe before:\n", orgData.head())



orgData = orgData.loc[:, orgData.apply(pd.Series.nunique) != 1]

print("\n\nDataframe after:\n", orgData.head())
# create a copy of a dataframe to work on

companyInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/CompanyInfo.csv')

companyInfo_copy = companyInfo.copy()



companyInfo_copy.head()
# sort pandas df by column

companyInfo.sort_values('People', ascending=False, inplace=True) # for multiple columns, pass in list of columns

companyInfo
# remove single row by index

companyInfo.drop(1, axis=0, inplace=True)

companyInfo
# get numpy values of entire DF

numpy_values = companyInfo.values

numpy_values
# extract 80% random rows from a column

companyInfo.Business.sample(frac=0.8)
# shuffle pandas rows randomly

companyInfo = pd.read_csv('/kaggle/input/all-pandas-operations-reference/CompanyInfo.csv')

sample = companyInfo.sample(frac=1)

sample
# save CSV to disk as a file, without saving index column

sample.to_csv('sample.csv', index=False)
'''

Powerful function to reduce RAM usgae, definitely use this for serious model building

reference: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65



parameters: 

props: DF to reduce

verbose: optional, default True

    Whether to print compression stats or not



returns:

df with reduced size

list of columns with NA filled in



usage: 

df, NAlist = reduce_mem_usage(df) # compression stats printed

df, _ = reduce_mem_usage(df, False) # compression stats NOT printed, ignore NA list



'''

def reduce_mem_usage(props, verbose):

    start_mem_usg = props.memory_usage().sum() / 1024**2

    

    if verbose:

        print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            if verbose:

                print("******************************")

                print("Column: ",col)

                print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            if verbose:

                print("dtype after: ",props[col].dtype)

                print("******************************")

    

    if verbose:

        # Print final result

        print("___MEMORY USAGE AFTER COMPLETION:___")

        mem_usg = props.memory_usage().sum() / 1024**2 

        print("Memory usage is: ",mem_usg," MB")

        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

        

    return props, NAlist
empData = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Employee_Data_1.csv')

empData, _ = reduce_mem_usage(empData, verbose=False)
# reset index of dataframe

countryInfo.reset_index(inplace=True)

countryInfo.head()
# use multiple datasets for slicing/dicing/further EDA



# read in one datafile of some countries

countryInfo1 = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Population_Language.csv')

countryInfo1.set_index('Country', inplace=True)



# read in one more datafile containing areas of above countries

countryInfo2 = pd.read_csv('/kaggle/input/all-pandas-operations-reference/Country_Area.csv')

countryInfo2.set_index('Country', inplace=True)



# calculate population density of all countries

countryInfo = countryInfo1.Population / countryInfo2.Area_in_million_sq_km

countryInfo
# merge/join/concatenate side by side dataframes. Dataframes should have the same indices

countryInfo = pd.concat([countryInfo1, countryInfo2], axis=1)

countryInfo.head()